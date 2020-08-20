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
Tensor _adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_adaptive_avg_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("_adaptive_avg_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
Tensor _adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_adaptive_avg_pool2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_adaptive_avg_pool2d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self);
}
Tensor & _addmv_impl_(Tensor & self, const Tensor & self2, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_addmv_impl_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("_addmv_impl_", std::vector<c10::IValue>({self, self2, mat, vec, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, self2, mat, vec, beta, alpha);
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
Tensor & _addr_out_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_addr", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("_addr_out", std::vector<c10::IValue>({out, self, vec1, vec2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, vec1, vec2, beta, alpha);
}
void _amp_non_finite_check_and_unscale_(Tensor & self, Tensor & found_inf, const Tensor & inv_scale) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_amp_non_finite_check_and_unscale_", "")
      .typed<void (Tensor &, Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_amp_non_finite_check_and_unscale_", std::vector<c10::IValue>({self, found_inf, inv_scale}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<void, Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, found_inf, inv_scale);
}
Tensor _amp_update_scale(Tensor & growth_tracker, const Tensor & current_scale, const Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_amp_update_scale", "")
      .typed<Tensor (Tensor &, const Tensor &, const Tensor &, double, double, int64_t)>();
  RECORD_FUNCTION("_amp_update_scale", std::vector<c10::IValue>({growth_tracker, current_scale, found_inf}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Tensor &, const Tensor &, const Tensor &, double, double, int64_t>(op, c10::DispatchKey::Profiler, growth_tracker, current_scale, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
}
Tensor & _baddbmm_mkl_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_baddbmm_mkl_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("_baddbmm_mkl_", std::vector<c10::IValue>({self, batch1, batch2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, batch1, batch2, beta, alpha);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t> _batch_norm_impl_index(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_batch_norm_impl_index", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool)>();
  RECORD_FUNCTION("_batch_norm_impl_index", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool>(op, c10::DispatchKey::Profiler, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}
std::tuple<Tensor,Tensor,Tensor> _batch_norm_impl_index_backward(int64_t impl_index, const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var_transform, bool train, double eps, std::array<bool,3> output_mask, const Tensor & reservedSpace) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_batch_norm_impl_index_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>, const Tensor &)>();
  RECORD_FUNCTION("_batch_norm_impl_index_backward", std::vector<c10::IValue>({input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, reservedSpace}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>, const Tensor &>(op, c10::DispatchKey::Profiler, impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace);
}
Tensor _bmm(const Tensor & self, const Tensor & mat2, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_bmm", "")
      .typed<Tensor (const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("_bmm", std::vector<c10::IValue>({self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, mat2, deterministic);
}
Tensor & _bmm_out_out(Tensor & out, const Tensor & self, const Tensor & mat2, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_bmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("_bmm_out", std::vector<c10::IValue>({out, self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, out, self, mat2, deterministic);
}
Tensor _cast_Byte(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Byte", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Byte", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, non_blocking);
}
Tensor _cast_Char(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Char", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Char", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, non_blocking);
}
Tensor _cast_Double(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Double", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Double", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, non_blocking);
}
Tensor _cast_Float(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Float", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Float", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
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
Tensor _cast_Long(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Long", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Long", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, non_blocking);
}
Tensor _cast_Short(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Short", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Short", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, non_blocking);
}
Tensor _cat(TensorList tensors, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cat", "")
      .typed<Tensor (TensorList, int64_t)>();
  RECORD_FUNCTION("_cat", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList, int64_t>(op, c10::DispatchKey::Profiler, tensors, dim);
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
Tensor _cdist_forward(const Tensor & x1, const Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cdist_forward", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, c10::optional<int64_t>)>();
  RECORD_FUNCTION("_cdist_forward", std::vector<c10::IValue>({x1, x2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, x1, x2, p, compute_mode);
}
Tensor _cholesky_helper(const Tensor & self, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cholesky_helper", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cholesky_helper", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, upper);
}
Tensor _cholesky_solve_helper(const Tensor & self, const Tensor & A, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cholesky_solve_helper", "")
      .typed<Tensor (const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("_cholesky_solve_helper", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, A, upper);
}
std::tuple<double,int64_t> _choose_qparams_per_tensor(const Tensor & self, bool reduce_range) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_choose_qparams_per_tensor", "")
      .typed<std::tuple<double,int64_t> (const Tensor &, bool)>();
  RECORD_FUNCTION("_choose_qparams_per_tensor", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<double,int64_t>, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, reduce_range);
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
std::tuple<Tensor,Tensor,Tensor> _convolution_double_backward(const Tensor & ggI, const Tensor & ggW, const Tensor & ggb, const Tensor & gO, const Tensor & weight, const Tensor & self, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_convolution_double_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool, std::array<bool,3>)>();
  RECORD_FUNCTION("_convolution_double_backward", std::vector<c10::IValue>({ggI, ggW, ggb, gO, weight, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool, std::array<bool,3>>(op, c10::DispatchKey::Profiler, ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);
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
std::tuple<Tensor,Tensor> _ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool zero_infinity) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_ctc_loss", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool)>();
  RECORD_FUNCTION("_ctc_loss", std::vector<c10::IValue>({log_probs, targets}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool>(op, c10::DispatchKey::Profiler, log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
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
Tensor _cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cudnn_init_dropout_state", "")
      .typed<Tensor (double, bool, int64_t, const TensorOptions &)>();
  RECORD_FUNCTION("_cudnn_init_dropout_state", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, double, bool, int64_t, const TensorOptions &>(op, c10::DispatchKey::Profiler, dropout, train, dropout_seed, options);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cudnn_rnn", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("_cudnn_rnn", std::vector<c10::IValue>({input, weight_buf, hx, cx, dropout_state}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> _cudnn_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cudnn_rnn_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> (const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,4>)>();
  RECORD_FUNCTION("_cudnn_rnn_backward", std::vector<c10::IValue>({input, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, dropout_state, reserve}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>>, const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,4>>(op, c10::DispatchKey::Profiler, input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
}
Tensor _cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cudnn_rnn_flatten_weight", "")
      .typed<Tensor (TensorList, int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool)>();
  RECORD_FUNCTION("_cudnn_rnn_flatten_weight", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList, int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight_arr, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
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
int64_t _cufft_get_plan_cache_size(int64_t device_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cufft_get_plan_cache_size", "")
      .typed<int64_t (int64_t)>();
  RECORD_FUNCTION("_cufft_get_plan_cache_size", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, int64_t>(op, c10::DispatchKey::Profiler, device_index);
}
void _cufft_set_plan_cache_max_size(int64_t device_index, int64_t max_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cufft_set_plan_cache_max_size", "")
      .typed<void (int64_t, int64_t)>();
  RECORD_FUNCTION("_cufft_set_plan_cache_max_size", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<void, int64_t, int64_t>(op, c10::DispatchKey::Profiler, device_index, max_size);
}
void _cummax_helper(const Tensor & self, Tensor & values, Tensor & indices, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cummax_helper", "")
      .typed<void (const Tensor &, Tensor &, Tensor &, int64_t)>();
  RECORD_FUNCTION("_cummax_helper", std::vector<c10::IValue>({self, values, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<void, const Tensor &, Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, values, indices, dim);
}
void _cummin_helper(const Tensor & self, Tensor & values, Tensor & indices, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cummin_helper", "")
      .typed<void (const Tensor &, Tensor &, Tensor &, int64_t)>();
  RECORD_FUNCTION("_cummin_helper", std::vector<c10::IValue>({self, values, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<void, const Tensor &, Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, values, indices, dim);
}
Tensor _cumprod(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cumprod", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("_cumprod", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
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
Tensor & _cumsum_out_out(Tensor & out, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cumsum", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_cumsum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, dim);
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
int64_t _dimV(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_dimV", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("_dimV", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
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
std::tuple<Tensor,Tensor,Tensor,Tensor> _embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_embedding_bag", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &, bool)>();
  RECORD_FUNCTION("_embedding_bag", std::vector<c10::IValue>({weight, indices, offsets, per_sample_weights}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &, bool>(op, c10::DispatchKey::Profiler, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
}
Tensor _embedding_bag_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_embedding_bag_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, bool, const Tensor &)>();
  RECORD_FUNCTION("_embedding_bag_backward", std::vector<c10::IValue>({grad, indices, offsets, offset2bag, bag_size, maximum_indices, per_sample_weights}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, bool, const Tensor &>(op, c10::DispatchKey::Profiler, grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights);
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
Tensor _embedding_bag_sparse_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const Tensor & per_sample_weights) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_embedding_bag_sparse_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const Tensor &)>();
  RECORD_FUNCTION("_embedding_bag_sparse_backward", std::vector<c10::IValue>({grad, indices, offsets, offset2bag, bag_size, per_sample_weights}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights);
}
Tensor _empty_affine_quantized(IntArrayRef size, const TensorOptions & options, double scale, int64_t zero_point, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_empty_affine_quantized", "")
      .typed<Tensor (IntArrayRef, const TensorOptions &, double, int64_t, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("_empty_affine_quantized", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const TensorOptions &, double, int64_t, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, size, options, scale, zero_point, memory_format);
}
Tensor _empty_per_channel_affine_quantized(IntArrayRef size, const Tensor & scales, const Tensor & zero_points, int64_t axis, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_empty_per_channel_affine_quantized", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, int64_t, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("_empty_per_channel_affine_quantized", std::vector<c10::IValue>({scales, zero_points}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, int64_t, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, size, scales, zero_points, axis, options, memory_format);
}
Tensor _euclidean_dist(const Tensor & x1, const Tensor & x2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_euclidean_dist", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_euclidean_dist", std::vector<c10::IValue>({x1, x2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, x1, x2);
}
Tensor _fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntArrayRef checked_signal_sizes, bool normalized, bool onesided, IntArrayRef output_sizes) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_fft_with_size", "")
      .typed<Tensor (const Tensor &, int64_t, bool, bool, bool, IntArrayRef, bool, bool, IntArrayRef)>();
  RECORD_FUNCTION("_fft_with_size", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool, bool, bool, IntArrayRef, bool, bool, IntArrayRef>(op, c10::DispatchKey::Profiler, self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
std::tuple<Tensor,Tensor> _fused_dropout(const Tensor & self, double p, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_fused_dropout", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("_fused_dropout", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, p, generator);
}
Tensor _gather_sparse_backward(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & grad) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_gather_sparse_backward", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_gather_sparse_backward", std::vector<c10::IValue>({self, index, grad}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, grad);
}
bool _has_compatible_shallow_copy_type(const Tensor & self, const Tensor & from) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_has_compatible_shallow_copy_type", "")
      .typed<bool (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_has_compatible_shallow_copy_type", std::vector<c10::IValue>({self, from}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, from);
}
Tensor & _index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_index_copy_", "")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_index_copy_", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, source);
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
Tensor _inverse_helper(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_inverse_helper", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("_inverse_helper", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Scalar _local_scalar_dense(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_local_scalar_dense", "")
      .typed<Scalar (const Tensor &)>();
  RECORD_FUNCTION("_local_scalar_dense", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor _log_softmax(const Tensor & self, int64_t dim, bool half_to_float) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_log_softmax", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("_log_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, half_to_float);
}
Tensor _log_softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_log_softmax_backward_data", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("_log_softmax_backward_data", std::vector<c10::IValue>({grad_output, output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, output, dim, self);
}
Tensor _logcumsumexp(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_logcumsumexp", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("_logcumsumexp", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor & _logcumsumexp_out_out(Tensor & out, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_logcumsumexp", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_logcumsumexp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, dim);
}
Tensor _lu_solve_helper(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_lu_solve_helper", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_lu_solve_helper", std::vector<c10::IValue>({self, LU_data, LU_pivots}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, LU_data, LU_pivots);
}
std::tuple<Tensor,Tensor,Tensor> _lu_with_info(const Tensor & self, bool pivot, bool check_errors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_lu_with_info", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool)>();
  RECORD_FUNCTION("_lu_with_info", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, pivot, check_errors);
}
Tensor _make_per_channel_quantized_tensor(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_make_per_channel_quantized_tensor", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_make_per_channel_quantized_tensor", std::vector<c10::IValue>({self, scale, zero_point}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, scale, zero_point, axis);
}
Tensor _make_per_tensor_quantized_tensor(const Tensor & self, double scale, int64_t zero_point) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_make_per_tensor_quantized_tensor", "")
      .typed<Tensor (const Tensor &, double, int64_t)>();
  RECORD_FUNCTION("_make_per_tensor_quantized_tensor", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, int64_t>(op, c10::DispatchKey::Profiler, self, scale, zero_point);
}
Tensor _masked_scale(const Tensor & self, const Tensor & mask, double scale) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_masked_scale", "")
      .typed<Tensor (const Tensor &, const Tensor &, double)>();
  RECORD_FUNCTION("_masked_scale", std::vector<c10::IValue>({self, mask}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double>(op, c10::DispatchKey::Profiler, self, mask, scale);
}
Tensor _mkldnn_reshape(const Tensor & self, IntArrayRef shape) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_mkldnn_reshape", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("_mkldnn_reshape", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, shape);
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
std::tuple<Tensor,Tensor> _mode(const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_mode", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("_mode", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> _mode_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_mode", "values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("_mode_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, values, indices, self, dim, keepdim);
}
Tensor _multinomial_alias_draw(const Tensor & J, const Tensor & q, int64_t num_samples, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_multinomial_alias_draw", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, c10::optional<Generator>)>();
  RECORD_FUNCTION("_multinomial_alias_draw", std::vector<c10::IValue>({J, q}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, J, q, num_samples, generator);
}
std::tuple<Tensor,Tensor> _multinomial_alias_setup(const Tensor & probs) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_multinomial_alias_setup", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &)>();
  RECORD_FUNCTION("_multinomial_alias_setup", std::vector<c10::IValue>({probs}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &>(op, c10::DispatchKey::Profiler, probs);
}
bool _nnpack_available() {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_nnpack_available", "")
      .typed<bool ()>();
  RECORD_FUNCTION("_nnpack_available", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool>(op, c10::DispatchKey::Profiler);
}
Tensor _nnpack_spatial_convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_nnpack_spatial_convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("_nnpack_spatial_convolution", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, input, weight, bias, padding, stride);
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
Tensor _nnpack_spatial_convolution_backward_weight(const Tensor & input, IntArrayRef weightsize, const Tensor & grad_output, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_nnpack_spatial_convolution_backward_weight", "")
      .typed<Tensor (const Tensor &, IntArrayRef, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("_nnpack_spatial_convolution_backward_weight", std::vector<c10::IValue>({input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, input, weightsize, grad_output, padding);
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
Tensor _pack_padded_sequence_backward(const Tensor & grad, IntArrayRef input_size, const Tensor & batch_sizes, bool batch_first) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_pack_padded_sequence_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, const Tensor &, bool)>();
  RECORD_FUNCTION("_pack_padded_sequence_backward", std::vector<c10::IValue>({grad, batch_sizes}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, const Tensor &, bool>(op, c10::DispatchKey::Profiler, grad, input_size, batch_sizes, batch_first);
}
std::tuple<Tensor,Tensor> _pad_packed_sequence(const Tensor & data, const Tensor & batch_sizes, bool batch_first, Scalar padding_value, int64_t total_length) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_pad_packed_sequence", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool, Scalar, int64_t)>();
  RECORD_FUNCTION("_pad_packed_sequence", std::vector<c10::IValue>({data, batch_sizes, padding_value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool, Scalar, int64_t>(op, c10::DispatchKey::Profiler, data, batch_sizes, batch_first, padding_value, total_length);
}
Tensor _pdist_backward(const Tensor & grad, const Tensor & self, double p, const Tensor & pdist) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_pdist_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, const Tensor &)>();
  RECORD_FUNCTION("_pdist_backward", std::vector<c10::IValue>({grad, self, pdist}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, const Tensor &>(op, c10::DispatchKey::Profiler, grad, self, p, pdist);
}
Tensor _pdist_forward(const Tensor & self, double p) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_pdist_forward", "")
      .typed<Tensor (const Tensor &, double)>();
  RECORD_FUNCTION("_pdist_forward", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double>(op, c10::DispatchKey::Profiler, self, p);
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
Tensor & _sobol_engine_scramble_(Tensor & self, const Tensor & ltm, int64_t dimension) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sobol_engine_scramble_", "")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_sobol_engine_scramble_", std::vector<c10::IValue>({self, ltm}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, ltm, dimension);
}
Tensor _softmax(const Tensor & self, int64_t dim, bool half_to_float) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_softmax", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, half_to_float);
}
Tensor _softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_softmax_backward_data", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("_softmax_backward_data", std::vector<c10::IValue>({grad_output, output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, output, dim, self);
}
std::tuple<Tensor,Tensor> _solve_helper(const Tensor & self, const Tensor & A) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_solve_helper", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_solve_helper", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, A);
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
Tensor _sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_coo_tensor_with_dims", "")
      .typed<Tensor (int64_t, int64_t, IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("_sparse_coo_tensor_with_dims", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, int64_t, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, sparse_dim, dense_dim, size, options);
}
Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_coo_tensor_with_dims_and_tensors", "")
      .typed<Tensor (int64_t, int64_t, IntArrayRef, const Tensor &, const Tensor &, const TensorOptions &)>();
  RECORD_FUNCTION("_sparse_coo_tensor_with_dims_and_tensors", std::vector<c10::IValue>({indices, values}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, int64_t, IntArrayRef, const Tensor &, const Tensor &, const TensorOptions &>(op, c10::DispatchKey::Profiler, sparse_dim, dense_dim, size, indices, values, options);
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
Tensor _sparse_sum_backward(const Tensor & grad, const Tensor & self, IntArrayRef dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_sum_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("_sparse_sum_backward", std::vector<c10::IValue>({grad, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad, self, dim);
}
Tensor _standard_gamma(const Tensor & self, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_standard_gamma", "")
      .typed<Tensor (const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("_standard_gamma", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, generator);
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
std::tuple<Tensor,Tensor> _symeig_helper(const Tensor & self, bool eigenvectors, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_symeig_helper", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool, bool)>();
  RECORD_FUNCTION("_symeig_helper", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, eigenvectors, upper);
}
Tensor _test_serialization_subcmul(const Tensor & self, const Tensor & other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_test_serialization_subcmul", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("_test_serialization_subcmul", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other, alpha);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_differentiable_gru_cell_backward(const Tensor & grad_hy, const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const Tensor & input_bias, const Tensor & hidden_bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_differentiable_gru_cell_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_thnn_differentiable_gru_cell_backward", std::vector<c10::IValue>({grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
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
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_fused_gru_cell_backward(const Tensor & grad_hy, const Tensor & workspace, bool has_bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_fused_gru_cell_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("_thnn_fused_gru_cell_backward", std::vector<c10::IValue>({grad_hy, workspace}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, grad_hy, workspace, has_bias);
}
std::tuple<Tensor,Tensor,Tensor> _thnn_fused_lstm_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & cx, const Tensor & input_bias, const Tensor & hidden_bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_fused_lstm_cell", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_thnn_fused_lstm_cell", std::vector<c10::IValue>({input_gates, hidden_gates, cx, input_bias, hidden_bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input_gates, hidden_gates, cx, input_bias, hidden_bias);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_fused_lstm_cell_backward(const Tensor & grad_hy, const Tensor & grad_cy, const Tensor & cx, const Tensor & cy, const Tensor & workspace, bool has_bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_fused_lstm_cell_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("_thnn_fused_lstm_cell_backward", std::vector<c10::IValue>({grad_hy, grad_cy, cx, cy, workspace}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, grad_hy, grad_cy, cx, cy, workspace, has_bias);
}
std::tuple<Tensor,Tensor> _triangular_solve_helper(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_triangular_solve_helper", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool, bool, bool)>();
  RECORD_FUNCTION("_triangular_solve_helper", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool, bool, bool>(op, c10::DispatchKey::Profiler, self, A, upper, transpose, unitriangular);
}
Tensor _trilinear(const Tensor & i1, const Tensor & i2, const Tensor & i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_trilinear", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("_trilinear", std::vector<c10::IValue>({i1, i2, i3}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
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
Tensor _unsafe_view(const Tensor & self, IntArrayRef size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_unsafe_view", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("_unsafe_view", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, size);
}
bool _use_cudnn_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_use_cudnn_ctc_loss", "")
      .typed<bool (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("_use_cudnn_ctc_loss", std::vector<c10::IValue>({log_probs, targets}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, log_probs, targets, input_lengths, target_lengths, blank);
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
int64_t _version(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_version", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("_version", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor _weight_norm(const Tensor & v, const Tensor & g, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_weight_norm", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_weight_norm", std::vector<c10::IValue>({v, g}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, v, g, dim);
}
std::tuple<Tensor,Tensor> _weight_norm_cuda_interface(const Tensor & v, const Tensor & g, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_weight_norm_cuda_interface", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_weight_norm_cuda_interface", std::vector<c10::IValue>({v, g}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, v, g, dim);
}
std::tuple<Tensor,Tensor> _weight_norm_cuda_interface_backward(const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_weight_norm_cuda_interface_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_weight_norm_cuda_interface_backward", std::vector<c10::IValue>({grad_w, saved_v, saved_g, saved_norms}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_w, saved_v, saved_g, saved_norms, dim);
}
std::tuple<Tensor,Tensor> _weight_norm_differentiable_backward(const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_weight_norm_differentiable_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_weight_norm_differentiable_backward", std::vector<c10::IValue>({grad_w, saved_v, saved_g, saved_norms}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_w, saved_v, saved_g, saved_norms, dim);
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
Tensor & abs_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::abs", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("abs_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
Tensor adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_avg_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
Tensor & adaptive_avg_pool2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_avg_pool2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, output_size);
}
Tensor adaptive_avg_pool3d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_avg_pool3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
Tensor adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("adaptive_avg_pool3d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self);
}
Tensor & adaptive_avg_pool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("adaptive_avg_pool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self);
}
Tensor & adaptive_avg_pool3d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_avg_pool3d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, output_size);
}
std::tuple<Tensor,Tensor> adaptive_max_pool1d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool1d", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_max_pool1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool2d", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_max_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
Tensor adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("adaptive_max_pool2d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, indices);
}
Tensor & adaptive_max_pool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("adaptive_max_pool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, indices);
}
std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool2d", "out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_max_pool2d_out", std::vector<c10::IValue>({out, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, indices, self, output_size);
}
std::tuple<Tensor,Tensor> adaptive_max_pool3d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool3d", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_max_pool3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
Tensor adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("adaptive_max_pool3d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, indices);
}
Tensor & adaptive_max_pool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("adaptive_max_pool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, indices);
}
std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_out_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool3d", "out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_max_pool3d_out", std::vector<c10::IValue>({out, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, indices, self, output_size);
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
Tensor & add_out_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::add", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("add_out", std::vector<c10::IValue>({out, self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other, alpha);
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
Tensor & addbmm_out_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addbmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addbmm_out", std::vector<c10::IValue>({out, self, batch1, batch2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, batch1, batch2, beta, alpha);
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
Tensor & addcdiv_out_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addcdiv", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("addcdiv_out", std::vector<c10::IValue>({out, self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, tensor1, tensor2, value);
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
Tensor & addcmul_out_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addcmul", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("addcmul_out", std::vector<c10::IValue>({out, self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, tensor1, tensor2, value);
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
Tensor & addmv_out_out(Tensor & out, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addmv", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addmv_out", std::vector<c10::IValue>({out, self, mat, vec, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, mat, vec, beta, alpha);
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
Tensor & addr_out_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addr", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addr_out", std::vector<c10::IValue>({out, self, vec1, vec2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, vec1, vec2, beta, alpha);
}
Tensor affine_grid_generator(const Tensor & theta, IntArrayRef size, bool align_corners) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::affine_grid_generator", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("affine_grid_generator", std::vector<c10::IValue>({theta}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, theta, size, align_corners);
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
Tensor argmax(const Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::argmax", "")
      .typed<Tensor (const Tensor &, c10::optional<int64_t>, bool)>();
  RECORD_FUNCTION("argmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<int64_t>, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor argmin(const Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::argmin", "")
      .typed<Tensor (const Tensor &, c10::optional<int64_t>, bool)>();
  RECORD_FUNCTION("argmin", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
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
Tensor & atan2_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan2", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("atan2_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & atan_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("atan_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
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
Tensor & atanh_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atanh", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("atanh_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor avg_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>();
  RECORD_FUNCTION("avg_pool1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool2d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor & avg_pool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor & avg_pool2d_out_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor avg_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool3d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor & avg_pool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor & avg_pool3d_out_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool3d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
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
Tensor batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool)>();
  RECORD_FUNCTION("batch_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool>(op, c10::DispatchKey::Profiler, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}
Tensor batch_norm_backward_elemt(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & weight, const Tensor & mean_dy, const Tensor & mean_dy_xmu) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_backward_elemt", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("batch_norm_backward_elemt", std::vector<c10::IValue>({grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu);
}
std::tuple<Tensor,Tensor,Tensor,Tensor> batch_norm_backward_reduce(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & weight, bool input_g, bool weight_g, bool bias_g) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_backward_reduce", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool, bool)>();
  RECORD_FUNCTION("batch_norm_backward_reduce", std::vector<c10::IValue>({grad_out, input, mean, invstd, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool, bool>(op, c10::DispatchKey::Profiler, grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
}
Tensor batch_norm_elemt(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & invstd, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_elemt", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double)>();
  RECORD_FUNCTION("batch_norm_elemt", std::vector<c10::IValue>({input, weight, bias, mean, invstd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double>(op, c10::DispatchKey::Profiler, input, weight, bias, mean, invstd, eps);
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
std::tuple<Tensor,Tensor> batch_norm_gather_stats_with_counts(const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & running_mean, const Tensor & running_var, double momentum, double eps, const Tensor & counts) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_gather_stats_with_counts", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, const Tensor &)>();
  RECORD_FUNCTION("batch_norm_gather_stats_with_counts", std::vector<c10::IValue>({input, mean, invstd, running_mean, running_var, counts}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, const Tensor &>(op, c10::DispatchKey::Profiler, input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}
std::tuple<Tensor,Tensor> batch_norm_stats(const Tensor & input, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_stats", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, double)>();
  RECORD_FUNCTION("batch_norm_stats", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, double>(op, c10::DispatchKey::Profiler, input, eps);
}
std::tuple<Tensor,Tensor> batch_norm_update_stats(const Tensor & input, const Tensor & running_mean, const Tensor & running_var, double momentum) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_update_stats", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, double)>();
  RECORD_FUNCTION("batch_norm_update_stats", std::vector<c10::IValue>({input, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, double>(op, c10::DispatchKey::Profiler, input, running_mean, running_var, momentum);
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
Tensor & bernoulli_out_out(Tensor & out, const Tensor & self, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bernoulli", "out")
      .typed<Tensor & (Tensor &, const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("bernoulli_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, self, generator);
}
Tensor bilinear(const Tensor & input1, const Tensor & input2, const Tensor & weight, const Tensor & bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bilinear", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bilinear", std::vector<c10::IValue>({input1, input2, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input1, input2, weight, bias);
}
Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("binary_cross_entropy", std::vector<c10::IValue>({self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, weight, reduction);
}
Tensor binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("binary_cross_entropy_backward", std::vector<c10::IValue>({grad_output, self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, weight, reduction);
}
Tensor & binary_cross_entropy_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("binary_cross_entropy_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, weight, reduction);
}
Tensor & binary_cross_entropy_out_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("binary_cross_entropy_out", std::vector<c10::IValue>({out, self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, weight, reduction);
}
Tensor binary_cross_entropy_with_logits(const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy_with_logits", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("binary_cross_entropy_with_logits", std::vector<c10::IValue>({self, target, weight, pos_weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, weight, pos_weight, reduction);
}
Tensor binary_cross_entropy_with_logits_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy_with_logits_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("binary_cross_entropy_with_logits_backward", std::vector<c10::IValue>({grad_output, self, target, weight, pos_weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, weight, pos_weight, reduction);
}
Tensor bincount(const Tensor & self, const Tensor & weights, int64_t minlength) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bincount", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("bincount", std::vector<c10::IValue>({self, weights}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, weights, minlength);
}
Tensor binomial(const Tensor & count, const Tensor & prob, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binomial", "")
      .typed<Tensor (const Tensor &, const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("binomial", std::vector<c10::IValue>({count, prob}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, count, prob, generator);
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
Tensor & bitwise_not_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_not", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bitwise_not_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
Tensor block_diag(TensorList tensors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::block_diag", "")
      .typed<Tensor (TensorList)>();
  RECORD_FUNCTION("block_diag", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList>(op, c10::DispatchKey::Profiler, tensors);
}
Tensor bmm(const Tensor & self, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bmm", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bmm", std::vector<c10::IValue>({self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, mat2);
}
Tensor & bmm_out_out(Tensor & out, const Tensor & self, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bmm_out", std::vector<c10::IValue>({out, self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, mat2);
}
std::vector<Tensor> broadcast_tensors(TensorList tensors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::broadcast_tensors", "")
      .typed<std::vector<Tensor> (TensorList)>();
  RECORD_FUNCTION("broadcast_tensors", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, TensorList>(op, c10::DispatchKey::Profiler, tensors);
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
Tensor & bucketize_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & boundaries, bool out_int32, bool right) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bucketize", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("bucketize_out", std::vector<c10::IValue>({out, self, boundaries}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, out, self, boundaries, out_int32, right);
}
bool can_cast(ScalarType from, ScalarType to) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::can_cast", "")
      .typed<bool (ScalarType, ScalarType)>();
  RECORD_FUNCTION("can_cast", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, ScalarType, ScalarType>(op, c10::DispatchKey::Profiler, from, to);
}
Tensor cartesian_prod(TensorList tensors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cartesian_prod", "")
      .typed<Tensor (TensorList)>();
  RECORD_FUNCTION("cartesian_prod", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList>(op, c10::DispatchKey::Profiler, tensors);
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
Tensor & cauchy_(Tensor & self, double median, double sigma, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cauchy_", "")
      .typed<Tensor & (Tensor &, double, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("cauchy_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, median, sigma, generator);
}
Tensor cdist(const Tensor & x1, const Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cdist", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, c10::optional<int64_t>)>();
  RECORD_FUNCTION("cdist", std::vector<c10::IValue>({x1, x2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, x1, x2, p, compute_mode);
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
Tensor & ceil_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ceil", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("ceil_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
Tensor chain_matmul(TensorList matrices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::chain_matmul", "")
      .typed<Tensor (TensorList)>();
  RECORD_FUNCTION("chain_matmul", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList>(op, c10::DispatchKey::Profiler, matrices);
}
Tensor channel_shuffle(const Tensor & self, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::channel_shuffle", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("channel_shuffle", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, groups);
}
Tensor cholesky(const Tensor & self, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("cholesky", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, upper);
}
Tensor cholesky_inverse(const Tensor & self, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky_inverse", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("cholesky_inverse", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, upper);
}
Tensor & cholesky_inverse_out_out(Tensor & out, const Tensor & self, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky_inverse", "out")
      .typed<Tensor & (Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("cholesky_inverse_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, out, self, upper);
}
Tensor & cholesky_out_out(Tensor & out, const Tensor & self, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky", "out")
      .typed<Tensor & (Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("cholesky_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, out, self, upper);
}
Tensor cholesky_solve(const Tensor & self, const Tensor & input2, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky_solve", "")
      .typed<Tensor (const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("cholesky_solve", std::vector<c10::IValue>({self, input2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, input2, upper);
}
Tensor & cholesky_solve_out_out(Tensor & out, const Tensor & self, const Tensor & input2, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky_solve", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("cholesky_solve_out", std::vector<c10::IValue>({out, self, input2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, out, self, input2, upper);
}
std::vector<Tensor> chunk(const Tensor & self, int64_t chunks, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::chunk", "")
      .typed<std::vector<Tensor> (const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("chunk", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, chunks, dim);
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
Tensor & clamp_max_out_out(Tensor & out, const Tensor & self, Scalar max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_max", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("clamp_max_out", std::vector<c10::IValue>({out, self, max}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, max);
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
Tensor & clamp_min_out_out(Tensor & out, const Tensor & self, Scalar min) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_min", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("clamp_min_out", std::vector<c10::IValue>({out, self, min}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, min);
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
Tensor coalesce(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::coalesce", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("coalesce", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor col2im(const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::col2im", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("col2im", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size, kernel_size, dilation, padding, stride);
}
Tensor col2im_backward(const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::col2im_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("col2im_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, kernel_size, dilation, padding, stride);
}
Tensor & col2im_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::col2im_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("col2im_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, kernel_size, dilation, padding, stride);
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
Tensor & conj_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conj", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("conj_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor constant_pad_nd(const Tensor & self, IntArrayRef pad, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::constant_pad_nd", "")
      .typed<Tensor (const Tensor &, IntArrayRef, Scalar)>();
  RECORD_FUNCTION("constant_pad_nd", std::vector<c10::IValue>({self, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, Scalar>(op, c10::DispatchKey::Profiler, self, pad, value);
}
Tensor contiguous(const Tensor & self, MemoryFormat memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::contiguous", "")
      .typed<Tensor (const Tensor &, MemoryFormat)>();
  RECORD_FUNCTION("contiguous", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, MemoryFormat>(op, c10::DispatchKey::Profiler, self, memory_format);
}
Tensor conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv1d", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("conv1d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, dilation, groups);
}
Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("conv2d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, dilation, groups);
}
Tensor conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv3d", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("conv3d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, dilation, groups);
}
Tensor conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv_tbc", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("conv_tbc", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, weight, bias, pad);
}
std::tuple<Tensor,Tensor,Tensor> conv_tbc_backward(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv_tbc_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("conv_tbc_backward", std::vector<c10::IValue>({self, input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, input, weight, bias, pad);
}
Tensor conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv_transpose1d", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef)>();
  RECORD_FUNCTION("conv_transpose1d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, output_padding, groups, dilation);
}
Tensor conv_transpose2d_input(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv_transpose2d", "input")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef)>();
  RECORD_FUNCTION("conv_transpose2d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, output_padding, groups, dilation);
}
Tensor conv_transpose3d_input(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv_transpose3d", "input")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef)>();
  RECORD_FUNCTION("conv_transpose3d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, output_padding, groups, dilation);
}
Tensor convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("convolution", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}
std::tuple<Tensor,Tensor,Tensor> convolution_backward_overrideable(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::convolution_backward_overrideable", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, std::array<bool,3>)>();
  RECORD_FUNCTION("convolution_backward_overrideable", std::vector<c10::IValue>({grad_output, input, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
}
Tensor convolution_overrideable(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::convolution_overrideable", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("convolution_overrideable", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
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
Tensor & cosh_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cosh", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("cosh_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor cosine_embedding_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cosine_embedding_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, double, int64_t)>();
  RECORD_FUNCTION("cosine_embedding_loss", std::vector<c10::IValue>({input1, input2, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, double, int64_t>(op, c10::DispatchKey::Profiler, input1, input2, target, margin, reduction);
}
Tensor cosine_similarity(const Tensor & x1, const Tensor & x2, int64_t dim, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cosine_similarity", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, double)>();
  RECORD_FUNCTION("cosine_similarity", std::vector<c10::IValue>({x1, x2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, double>(op, c10::DispatchKey::Profiler, x1, x2, dim, eps);
}
Tensor cross(const Tensor & self, const Tensor & other, c10::optional<int64_t> dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cross", "")
      .typed<Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>)>();
  RECORD_FUNCTION("cross", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, other, dim);
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
Tensor cudnn_affine_grid_generator(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_affine_grid_generator", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("cudnn_affine_grid_generator", std::vector<c10::IValue>({theta}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, theta, N, C, H, W);
}
Tensor cudnn_affine_grid_generator_backward(const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_affine_grid_generator_backward", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("cudnn_affine_grid_generator_backward", std::vector<c10::IValue>({grad}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, grad, N, C, H, W);
}
std::tuple<Tensor,Tensor,Tensor,Tensor> cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_batch_norm", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>();
  RECORD_FUNCTION("cudnn_batch_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(op, c10::DispatchKey::Profiler, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}
std::tuple<Tensor,Tensor,Tensor> cudnn_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon, const Tensor & reserveSpace) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_batch_norm_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, const Tensor &)>();
  RECORD_FUNCTION("cudnn_batch_norm_backward", std::vector<c10::IValue>({input, grad_output, weight, running_mean, running_var, save_mean, save_var, reserveSpace}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, const Tensor &>(op, c10::DispatchKey::Profiler, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace);
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
Tensor cudnn_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution_backward_weight", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("cudnn_convolution_backward_weight", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
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
Tensor cudnn_convolution_transpose_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution_transpose_backward_weight", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("cudnn_convolution_transpose_backward_weight", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor cudnn_grid_sampler(const Tensor & self, const Tensor & grid) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_grid_sampler", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("cudnn_grid_sampler", std::vector<c10::IValue>({self, grid}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, grid);
}
std::tuple<Tensor,Tensor> cudnn_grid_sampler_backward(const Tensor & self, const Tensor & grid, const Tensor & grad_output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_grid_sampler_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("cudnn_grid_sampler_backward", std::vector<c10::IValue>({self, grid, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, grid, grad_output);
}
bool cudnn_is_acceptable(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_is_acceptable", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("cudnn_is_acceptable", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
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
Tensor & deg2rad_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::deg2rad", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("deg2rad_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
int64_t dense_dim(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dense_dim", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("dense_dim", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
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
Tensor diag(const Tensor & self, int64_t diagonal) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::diag", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("diag", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, diagonal);
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
Tensor diagflat(const Tensor & self, int64_t offset) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::diagflat", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("diagflat", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, offset);
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
Tensor & digamma_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::digamma", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("digamma_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor dist(const Tensor & self, const Tensor & other, Scalar p) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dist", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("dist", std::vector<c10::IValue>({self, other, p}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other, p);
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
std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eig", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool)>();
  RECORD_FUNCTION("eig", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, eigenvectors);
}
std::tuple<Tensor &,Tensor &> eig_out_e(Tensor & e, Tensor & v, const Tensor & self, bool eigenvectors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eig", "e")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("eig_out", std::vector<c10::IValue>({e, v, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, e, v, self, eigenvectors);
}
Tensor einsum(std::string equation, TensorList tensors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::einsum", "")
      .typed<Tensor (std::string, TensorList)>();
  RECORD_FUNCTION("einsum", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, std::string, TensorList>(op, c10::DispatchKey::Profiler, equation, tensors);
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
Tensor elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::elu_backward", "")
      .typed<Tensor (const Tensor &, Scalar, Scalar, Scalar, const Tensor &)>();
  RECORD_FUNCTION("elu_backward", std::vector<c10::IValue>({grad_output, alpha, scale, input_scale, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, alpha, scale, input_scale, output);
}
Tensor & elu_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::elu_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar, Scalar, const Tensor &)>();
  RECORD_FUNCTION("elu_backward_out", std::vector<c10::IValue>({grad_input, grad_output, alpha, scale, input_scale, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, alpha, scale, input_scale, output);
}
Tensor & elu_out_out(Tensor & out, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::elu", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("elu_out", std::vector<c10::IValue>({out, self, alpha, scale, input_scale}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, alpha, scale, input_scale);
}
Tensor embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::embedding", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, bool, bool)>();
  RECORD_FUNCTION("embedding", std::vector<c10::IValue>({weight, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight, indices, padding_idx, scale_grad_by_freq, sparse);
}
Tensor embedding_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::embedding_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, int64_t, bool, bool)>();
  RECORD_FUNCTION("embedding_backward", std::vector<c10::IValue>({grad, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
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
Tensor & embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::embedding_renorm_", "")
      .typed<Tensor & (Tensor &, const Tensor &, double, double)>();
  RECORD_FUNCTION("embedding_renorm_", std::vector<c10::IValue>({self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, double, double>(op, c10::DispatchKey::Profiler, self, indices, max_norm, norm_type);
}
Tensor embedding_sparse_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::embedding_sparse_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  RECORD_FUNCTION("embedding_sparse_backward", std::vector<c10::IValue>({grad, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Profiler, grad, indices, num_weights, padding_idx, scale_grad_by_freq);
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
Tensor empty_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty_like", "")
      .typed<Tensor (const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("empty_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, options, memory_format);
}
Tensor empty_meta(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty_meta", "")
      .typed<Tensor (IntArrayRef, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("empty_meta", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, size, options, memory_format);
}
Tensor & empty_out_out(Tensor & out, IntArrayRef size, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty", "out")
      .typed<Tensor & (Tensor &, IntArrayRef, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("empty_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, out, size, memory_format);
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
bool equal(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::equal", "")
      .typed<bool (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("equal", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
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
Tensor & erfinv_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erfinv", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("erfinv_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
Tensor & exp_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::exp", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("exp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor expand(const Tensor & self, IntArrayRef size, bool implicit) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::expand", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("expand", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, size, implicit);
}
Tensor expand_as(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::expand_as", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("expand_as", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
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
Tensor & expm1_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::expm1", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("expm1_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor & exponential_(Tensor & self, double lambd, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::exponential_", "")
      .typed<Tensor & (Tensor &, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("exponential_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, lambd, generator);
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
Tensor fake_quantize_per_channel_affine(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fake_quantize_per_channel_affine", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("fake_quantize_per_channel_affine", std::vector<c10::IValue>({self, scale, zero_point}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, scale, zero_point, axis, quant_min, quant_max);
}
Tensor fake_quantize_per_channel_affine_backward(const Tensor & grad, const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fake_quantize_per_channel_affine_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("fake_quantize_per_channel_affine_backward", std::vector<c10::IValue>({grad, self, scale, zero_point}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, grad, self, scale, zero_point, axis, quant_min, quant_max);
}
Tensor fake_quantize_per_tensor_affine(const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fake_quantize_per_tensor_affine", "")
      .typed<Tensor (const Tensor &, double, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("fake_quantize_per_tensor_affine", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, scale, zero_point, quant_min, quant_max);
}
Tensor fake_quantize_per_tensor_affine_backward(const Tensor & grad, const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fake_quantize_per_tensor_affine_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("fake_quantize_per_tensor_affine_backward", std::vector<c10::IValue>({grad, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, grad, self, scale, zero_point, quant_min, quant_max);
}
Tensor fbgemm_linear_fp16_weight(const Tensor & input, const Tensor & packed_weight, const Tensor & bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_linear_fp16_weight", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("fbgemm_linear_fp16_weight", std::vector<c10::IValue>({input, packed_weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, packed_weight, bias);
}
Tensor fbgemm_linear_fp16_weight_fp32_activation(const Tensor & input, const Tensor & packed_weight, const Tensor & bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_linear_fp16_weight_fp32_activation", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("fbgemm_linear_fp16_weight_fp32_activation", std::vector<c10::IValue>({input, packed_weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, packed_weight, bias);
}
Tensor fbgemm_linear_int8_weight(const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, Scalar weight_scale, Scalar weight_zero_point, const Tensor & bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_linear_int8_weight", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>();
  RECORD_FUNCTION("fbgemm_linear_int8_weight", std::vector<c10::IValue>({input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
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
Tensor fbgemm_pack_gemm_matrix_fp16(const Tensor & input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_pack_gemm_matrix_fp16", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("fbgemm_pack_gemm_matrix_fp16", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, input);
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
Tensor fft(const Tensor & self, int64_t signal_ndim, bool normalized) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fft", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("fft", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, signal_ndim, normalized);
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
Tensor & fill_diagonal_(Tensor & self, Scalar fill_value, bool wrap) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fill_diagonal_", "")
      .typed<Tensor & (Tensor &, Scalar, bool)>();
  RECORD_FUNCTION("fill_diagonal_", std::vector<c10::IValue>({self, fill_value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, bool>(op, c10::DispatchKey::Profiler, self, fill_value, wrap);
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
Tensor flip(const Tensor & self, IntArrayRef dims) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::flip", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("flip", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, dims);
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
Tensor & floor_divide_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor_divide", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("floor_divide_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & floor_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("floor_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
Tensor & frac_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::frac", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("frac_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
std::tuple<Tensor,Tensor> fractional_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool2d", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool2d", std::vector<c10::IValue>({self, random_samples}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, self, kernel_size, output_size, random_samples);
}
Tensor fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool2d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, kernel_size, output_size, indices);
}
Tensor & fractional_max_pool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, kernel_size, output_size, indices);
}
std::tuple<Tensor &,Tensor &> fractional_max_pool2d_out_output(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool2d", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool2d_out", std::vector<c10::IValue>({output, indices, self, random_samples}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, output, indices, self, kernel_size, output_size, random_samples);
}
std::tuple<Tensor,Tensor> fractional_max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool3d", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool3d", std::vector<c10::IValue>({self, random_samples}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, self, kernel_size, output_size, random_samples);
}
Tensor fractional_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool3d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, kernel_size, output_size, indices);
}
Tensor & fractional_max_pool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
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
Tensor & frobenius_norm_out_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::frobenius_norm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("frobenius_norm_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim);
}
Tensor from_file(std::string filename, c10::optional<bool> shared, c10::optional<int64_t> size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::from_file", "")
      .typed<Tensor (std::string, c10::optional<bool>, c10::optional<int64_t>, const TensorOptions &)>();
  RECORD_FUNCTION("from_file", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, std::string, c10::optional<bool>, c10::optional<int64_t>, const TensorOptions &>(op, c10::DispatchKey::Profiler, filename, shared, size, options);
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
Tensor full_like(const Tensor & self, Scalar fill_value, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::full_like", "")
      .typed<Tensor (const Tensor &, Scalar, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("full_like", std::vector<c10::IValue>({self, fill_value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, fill_value, options, memory_format);
}
Tensor & full_out_out(Tensor & out, IntArrayRef size, Scalar fill_value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::full", "out")
      .typed<Tensor & (Tensor &, IntArrayRef, Scalar)>();
  RECORD_FUNCTION("full_out", std::vector<c10::IValue>({out, fill_value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, Scalar>(op, c10::DispatchKey::Profiler, out, size, fill_value);
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
Tensor gelu(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gelu", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("gelu", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor gelu_backward(const Tensor & grad, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gelu_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("gelu_backward", std::vector<c10::IValue>({grad, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad, self);
}
Tensor & geometric_(Tensor & self, double p, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::geometric_", "")
      .typed<Tensor & (Tensor &, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("geometric_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, p, generator);
}
std::tuple<Tensor,Tensor> geqrf(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::geqrf", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &)>();
  RECORD_FUNCTION("geqrf", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor &,Tensor &> geqrf_out_a(Tensor & a, Tensor & tau, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::geqrf", "a")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &)>();
  RECORD_FUNCTION("geqrf_out", std::vector<c10::IValue>({a, tau, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, a, tau, self);
}
Tensor ger(const Tensor & self, const Tensor & vec2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ger", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("ger", std::vector<c10::IValue>({self, vec2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, vec2);
}
Tensor & ger_out_out(Tensor & out, const Tensor & self, const Tensor & vec2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ger", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("ger_out", std::vector<c10::IValue>({out, self, vec2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, vec2);
}
Tensor glu(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::glu", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("glu", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::glu_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("glu_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, dim);
}
Tensor & glu_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::glu_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("glu_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, dim);
}
Tensor & glu_out_out(Tensor & out, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::glu", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("glu_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, dim);
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
std::tuple<Tensor,Tensor> grid_sampler_3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::grid_sampler_3d_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  RECORD_FUNCTION("grid_sampler_3d_backward", std::vector<c10::IValue>({grad_output, input, grid}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Profiler, grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}
Tensor group_norm(const Tensor & input, int64_t num_groups, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enabled) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::group_norm", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &, double, bool)>();
  RECORD_FUNCTION("group_norm", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &, double, bool>(op, c10::DispatchKey::Profiler, input, num_groups, weight, bias, eps, cudnn_enabled);
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
Tensor gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gru_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("gru_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh);
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
Tensor hardshrink_backward(const Tensor & grad_out, const Tensor & self, Scalar lambd) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardshrink_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("hardshrink_backward", std::vector<c10::IValue>({grad_out, self, lambd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, grad_out, self, lambd);
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
Tensor hardsigmoid_backward(const Tensor & grad_output, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardsigmoid_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("hardsigmoid_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self);
}
Tensor & hardsigmoid_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardsigmoid", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("hardsigmoid_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
Tensor hardswish_backward(const Tensor & grad_output, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardswish_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("hardswish_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self);
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
Tensor hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardtanh_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("hardtanh_backward", std::vector<c10::IValue>({grad_output, self, min_val, max_val}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, grad_output, self, min_val, max_val);
}
Tensor & hardtanh_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardtanh_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("hardtanh_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, min_val, max_val}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, min_val, max_val);
}
Tensor & hardtanh_out_out(Tensor & out, const Tensor & self, Scalar min_val, Scalar max_val) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardtanh", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("hardtanh_out", std::vector<c10::IValue>({out, self, min_val, max_val}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, min_val, max_val);
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
Tensor & histc_out_out(Tensor & out, const Tensor & self, int64_t bins, Scalar min, Scalar max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::histc", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, Scalar, Scalar)>();
  RECORD_FUNCTION("histc_out", std::vector<c10::IValue>({out, self, min, max}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, bins, min, max);
}
Tensor hspmm(const Tensor & mat1, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hspmm", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("hspmm", std::vector<c10::IValue>({mat1, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, mat1, mat2);
}
Tensor & hspmm_out_out(Tensor & out, const Tensor & mat1, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hspmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("hspmm_out", std::vector<c10::IValue>({out, mat1, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, mat1, mat2);
}
Tensor ifft(const Tensor & self, int64_t signal_ndim, bool normalized) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ifft", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("ifft", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, signal_ndim, normalized);
}
Tensor im2col(const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::im2col", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("im2col", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, kernel_size, dilation, padding, stride);
}
Tensor im2col_backward(const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::im2col_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("im2col_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, input_size, kernel_size, dilation, padding, stride);
}
Tensor & im2col_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::im2col_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("im2col_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, input_size, kernel_size, dilation, padding, stride);
}
Tensor & im2col_out_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::im2col", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("im2col_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, kernel_size, dilation, padding, stride);
}
Tensor imag(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::imag", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("imag", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor index_Tensor(const Tensor & self, TensorList indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index", "Tensor")
      .typed<Tensor (const Tensor &, TensorList)>();
  RECORD_FUNCTION("index", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, TensorList>(op, c10::DispatchKey::Profiler, self, indices);
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
Tensor inverse(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::inverse", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("inverse", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & inverse_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::inverse", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("inverse_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor irfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::irfft", "")
      .typed<Tensor (const Tensor &, int64_t, bool, bool, IntArrayRef)>();
  RECORD_FUNCTION("irfft", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool, bool, IntArrayRef>(op, c10::DispatchKey::Profiler, self, signal_ndim, normalized, onesided, signal_sizes);
}
bool is_coalesced(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_coalesced", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_coalesced", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_complex(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_complex", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_complex", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_distributed(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_distributed", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_distributed", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_floating_point(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_floating_point", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_floating_point", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_leaf(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_leaf", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_leaf", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_nonzero(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_nonzero", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_nonzero", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_pinned(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_pinned", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_pinned", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_same_size(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_same_size", "")
      .typed<bool (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("is_same_size", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
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
bool is_vulkan_available() {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_vulkan_available", "")
      .typed<bool ()>();
  RECORD_FUNCTION("is_vulkan_available", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool>(op, c10::DispatchKey::Profiler);
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
Tensor istft(const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool center, bool normalized, bool onesided, c10::optional<int64_t> length) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::istft", "")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("istft", std::vector<c10::IValue>({self, window}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, n_fft, hop_length, win_length, window, center, normalized, onesided, length);
}
Scalar item(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::item", "")
      .typed<Scalar (const Tensor &)>();
  RECORD_FUNCTION("item", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor kl_div(const Tensor & self, const Tensor & target, int64_t reduction, bool log_target) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::kl_div", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("kl_div", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, target, reduction, log_target);
}
Tensor kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, bool log_target) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::kl_div_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("kl_div_backward", std::vector<c10::IValue>({grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, grad_output, self, target, reduction, log_target);
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
Tensor l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::l1_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("l1_loss", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, reduction);
}
Tensor l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::l1_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("l1_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, reduction);
}
Tensor & l1_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::l1_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("l1_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, reduction);
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
Tensor leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope, bool self_is_result) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::leaky_relu_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar, bool)>();
  RECORD_FUNCTION("leaky_relu_backward", std::vector<c10::IValue>({grad_output, self, negative_slope}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar, bool>(op, c10::DispatchKey::Profiler, grad_output, self, negative_slope, self_is_result);
}
Tensor & leaky_relu_out_out(Tensor & out, const Tensor & self, Scalar negative_slope) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::leaky_relu", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("leaky_relu_out", std::vector<c10::IValue>({out, self, negative_slope}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, negative_slope);
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
Tensor & lgamma_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lgamma", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lgamma_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor linear(const Tensor & input, const Tensor & weight, const Tensor & bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::linear", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("linear", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, weight, bias);
}
Tensor linspace(Scalar start, Scalar end, int64_t steps, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::linspace", "")
      .typed<Tensor (Scalar, Scalar, int64_t, const TensorOptions &)>();
  RECORD_FUNCTION("linspace", std::vector<c10::IValue>({start, end}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Scalar, Scalar, int64_t, const TensorOptions &>(op, c10::DispatchKey::Profiler, start, end, steps, options);
}
Tensor & linspace_out_out(Tensor & out, Scalar start, Scalar end, int64_t steps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::linspace", "out")
      .typed<Tensor & (Tensor &, Scalar, Scalar, int64_t)>();
  RECORD_FUNCTION("linspace_out", std::vector<c10::IValue>({out, start, end}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, int64_t>(op, c10::DispatchKey::Profiler, out, start, end, steps);
}
Tensor log(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("log", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
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
Tensor & log2_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log2", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("log2_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor & log_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("log_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
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
Tensor log_sigmoid(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("log_sigmoid", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("log_sigmoid_backward", std::vector<c10::IValue>({grad_output, self, buffer}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, buffer);
}
Tensor & log_sigmoid_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("log_sigmoid_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, buffer}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, buffer);
}
std::tuple<Tensor,Tensor> log_sigmoid_forward(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid_forward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &)>();
  RECORD_FUNCTION("log_sigmoid_forward", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor &,Tensor &> log_sigmoid_forward_out_output(Tensor & output, Tensor & buffer, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &)>();
  RECORD_FUNCTION("log_sigmoid_forward_out", std::vector<c10::IValue>({output, buffer, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, output, buffer, self);
}
Tensor & log_sigmoid_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("log_sigmoid_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
Tensor logdet(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logdet", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("logdet", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
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
Tensor & logical_and_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_and", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_and_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
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
Tensor & logical_not_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_not", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_not_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
Tensor & logical_or_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_or", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_or_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
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
Tensor & logical_xor_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_xor", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_xor_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor logspace(Scalar start, Scalar end, int64_t steps, double base, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logspace", "")
      .typed<Tensor (Scalar, Scalar, int64_t, double, const TensorOptions &)>();
  RECORD_FUNCTION("logspace", std::vector<c10::IValue>({start, end}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Scalar, Scalar, int64_t, double, const TensorOptions &>(op, c10::DispatchKey::Profiler, start, end, steps, base, options);
}
Tensor & logspace_out_out(Tensor & out, Scalar start, Scalar end, int64_t steps, double base) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logspace", "out")
      .typed<Tensor & (Tensor &, Scalar, Scalar, int64_t, double)>();
  RECORD_FUNCTION("logspace_out", std::vector<c10::IValue>({out, start, end}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, int64_t, double>(op, c10::DispatchKey::Profiler, out, start, end, steps, base);
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
std::tuple<Tensor,Tensor> lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lstm_cell", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lstm_cell", std::vector<c10::IValue>({input, w_ih, w_hh, b_ih, b_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh);
}
std::tuple<Tensor,Tensor> lstsq(const Tensor & self, const Tensor & A) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lstsq", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lstsq", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, A);
}
std::tuple<Tensor &,Tensor &> lstsq_out_X(Tensor & X, Tensor & qr, const Tensor & self, const Tensor & A) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lstsq", "X")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lstsq_out", std::vector<c10::IValue>({X, qr, self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, X, qr, self, A);
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
Tensor & lu_solve_out_out(Tensor & out, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lu_solve", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lu_solve_out", std::vector<c10::IValue>({out, self, LU_data, LU_pivots}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, LU_data, LU_pivots);
}
Tensor margin_ranking_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::margin_ranking_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, double, int64_t)>();
  RECORD_FUNCTION("margin_ranking_loss", std::vector<c10::IValue>({input1, input2, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, double, int64_t>(op, c10::DispatchKey::Profiler, input1, input2, target, margin, reduction);
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
Tensor masked_select(const Tensor & self, const Tensor & mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::masked_select", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("masked_select", std::vector<c10::IValue>({self, mask}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, mask);
}
Tensor & masked_select_out_out(Tensor & out, const Tensor & self, const Tensor & mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::masked_select", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("masked_select_out", std::vector<c10::IValue>({out, self, mask}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, mask);
}
Tensor matmul(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::matmul", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("matmul", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & matmul_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::matmul", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("matmul_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor matrix_power(const Tensor & self, int64_t n) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::matrix_power", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("matrix_power", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, n);
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
Tensor max_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
std::tuple<Tensor,Tensor> max_pool1d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool1d_with_indices", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool1d_with_indices", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
std::tuple<Tensor,Tensor> max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool2d_with_indices", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool2d_with_indices", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool2d_with_indices_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>();
  RECORD_FUNCTION("max_pool2d_with_indices_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
Tensor & max_pool2d_with_indices_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool2d_with_indices_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>();
  RECORD_FUNCTION("max_pool2d_with_indices_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool2d_with_indices", "out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool2d_with_indices_out", std::vector<c10::IValue>({out, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
std::tuple<Tensor,Tensor> max_pool3d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool3d_with_indices", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool3d_with_indices", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor max_pool3d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool3d_with_indices_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>();
  RECORD_FUNCTION("max_pool3d_with_indices_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
Tensor & max_pool3d_with_indices_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool3d_with_indices_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>();
  RECORD_FUNCTION("max_pool3d_with_indices_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
std::tuple<Tensor &,Tensor &> max_pool3d_with_indices_out_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool3d_with_indices", "out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool3d_with_indices_out", std::vector<c10::IValue>({out, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor max_unpool2d(const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool2d", std::vector<c10::IValue>({self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, indices, output_size);
}
Tensor max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool2d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, indices, output_size);
}
Tensor & max_unpool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, indices, output_size);
}
Tensor & max_unpool2d_out_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool2d_out", std::vector<c10::IValue>({out, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, indices, output_size);
}
Tensor max_unpool3d(const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool3d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool3d", std::vector<c10::IValue>({self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, indices, output_size, stride, padding);
}
Tensor max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool3d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, indices, output_size, stride, padding);
}
Tensor & max_unpool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, indices, output_size, stride, padding);
}
Tensor & max_unpool3d_out_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool3d_out", std::vector<c10::IValue>({out, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, indices, output_size, stride, padding);
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
std::tuple<Tensor,Tensor,Tensor> miopen_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_batch_norm", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>();
  RECORD_FUNCTION("miopen_batch_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(op, c10::DispatchKey::Profiler, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}
std::tuple<Tensor,Tensor,Tensor> miopen_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_batch_norm_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double)>();
  RECORD_FUNCTION("miopen_batch_norm_backward", std::vector<c10::IValue>({input, grad_output, weight, running_mean, running_var, save_mean, save_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double>(op, c10::DispatchKey::Profiler, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
}
Tensor miopen_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_convolution", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
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
Tensor miopen_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_backward_weight", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_convolution_backward_weight", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor miopen_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_transpose", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_convolution_transpose", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
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
Tensor miopen_convolution_transpose_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_transpose_backward_weight", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_convolution_transpose_backward_weight", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor miopen_depthwise_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_depthwise_convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_depthwise_convolution", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
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
Tensor miopen_depthwise_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_depthwise_convolution_backward_weight", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_depthwise_convolution_backward_weight", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> miopen_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_rnn", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("miopen_rnn", std::vector<c10::IValue>({input, hx, cx, dropout_state}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> miopen_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_rnn_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> (const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,4>)>();
  RECORD_FUNCTION("miopen_rnn_backward", std::vector<c10::IValue>({input, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, dropout_state, reserve}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>>, const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,4>>(op, c10::DispatchKey::Profiler, input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
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
Tensor mkldnn_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_max_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("mkldnn_max_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor mkldnn_reorder_conv2d_weight(const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_reorder_conv2d_weight", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("mkldnn_reorder_conv2d_weight", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, self, padding, stride, dilation, groups);
}
Tensor mm(const Tensor & self, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mm", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("mm", std::vector<c10::IValue>({self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, mat2);
}
Tensor & mm_out_out(Tensor & out, const Tensor & self, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("mm_out", std::vector<c10::IValue>({out, self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, mat2);
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
Tensor mse_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mse_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("mse_loss", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, reduction);
}
Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mse_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("mse_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, reduction);
}
Tensor & mse_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mse_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("mse_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, reduction);
}
Tensor & mse_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mse_loss", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("mse_loss_out", std::vector<c10::IValue>({out, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, reduction);
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
Tensor & mul_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mul", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("mul_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multi_margin_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multi_margin_loss", std::vector<c10::IValue>({self, target, p, margin, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, p, margin, weight, reduction);
}
Tensor multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multi_margin_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multi_margin_loss_backward", std::vector<c10::IValue>({grad_output, self, target, p, margin, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, p, margin, weight, reduction);
}
Tensor & multi_margin_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multi_margin_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multi_margin_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, p, margin, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, p, margin, weight, reduction);
}
Tensor & multi_margin_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multi_margin_loss", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multi_margin_loss_out", std::vector<c10::IValue>({out, self, target, p, margin, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, p, margin, weight, reduction);
}
Tensor multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multilabel_margin_loss", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, reduction);
}
Tensor multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("multilabel_margin_loss_backward", std::vector<c10::IValue>({grad_output, self, target, is_target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, target, reduction, is_target);
}
Tensor & multilabel_margin_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("multilabel_margin_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, is_target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, reduction, is_target);
}
std::tuple<Tensor,Tensor> multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss_forward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multilabel_margin_loss_forward", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, reduction);
}
std::tuple<Tensor &,Tensor &> multilabel_margin_loss_forward_out_output(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multilabel_margin_loss_forward_out", std::vector<c10::IValue>({output, is_target, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, output, is_target, self, target, reduction);
}
Tensor & multilabel_margin_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multilabel_margin_loss_out", std::vector<c10::IValue>({out, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, reduction);
}
Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multinomial", "")
      .typed<Tensor (const Tensor &, int64_t, bool, c10::optional<Generator>)>();
  RECORD_FUNCTION("multinomial", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, num_samples, replacement, generator);
}
Tensor & multinomial_out_out(Tensor & out, const Tensor & self, int64_t num_samples, bool replacement, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multinomial", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, bool, c10::optional<Generator>)>();
  RECORD_FUNCTION("multinomial_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, bool, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, self, num_samples, replacement, generator);
}
Tensor mv(const Tensor & self, const Tensor & vec) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mv", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("mv", std::vector<c10::IValue>({self, vec}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, vec);
}
Tensor & mv_out_out(Tensor & out, const Tensor & self, const Tensor & vec) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mv", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("mv_out", std::vector<c10::IValue>({out, self, vec}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, vec);
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
Tensor narrow_copy(const Tensor & self, int64_t dim, int64_t start, int64_t length) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::narrow_copy", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("narrow_copy", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, dim, start, length);
}
std::tuple<Tensor,Tensor,Tensor> native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_batch_norm", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>();
  RECORD_FUNCTION("native_batch_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(op, c10::DispatchKey::Profiler, input, weight, bias, running_mean, running_var, training, momentum, eps);
}
std::tuple<Tensor,Tensor,Tensor> native_batch_norm_backward(const Tensor & grad_out, const Tensor & input, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_invstd, bool train, double eps, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_batch_norm_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>)>();
  RECORD_FUNCTION("native_batch_norm_backward", std::vector<c10::IValue>({grad_out, input, weight, running_mean, running_var, save_mean, save_invstd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
}
std::tuple<Tensor &,Tensor &,Tensor &> native_batch_norm_out_out(Tensor & out, Tensor & save_mean, Tensor & save_invstd, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_batch_norm", "out")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>();
  RECORD_FUNCTION("native_batch_norm_out", std::vector<c10::IValue>({out, save_mean, save_invstd, input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(op, c10::DispatchKey::Profiler, out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
}
std::tuple<Tensor,Tensor,Tensor> native_group_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_group_norm", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, int64_t, double)>();
  RECORD_FUNCTION("native_group_norm", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, int64_t, double>(op, c10::DispatchKey::Profiler, input, weight, bias, N, C, HxW, group, eps);
}
std::tuple<Tensor,Tensor,Tensor> native_group_norm_backward(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & rstd, const Tensor & weight, int64_t N, int64_t C, int64_t HxW, int64_t group, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_group_norm_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, int64_t, std::array<bool,3>)>();
  RECORD_FUNCTION("native_group_norm_backward", std::vector<c10::IValue>({grad_out, input, mean, rstd, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, int64_t, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
}
std::tuple<Tensor,Tensor,Tensor> native_layer_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t M, int64_t N, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_layer_norm", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double)>();
  RECORD_FUNCTION("native_layer_norm", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double>(op, c10::DispatchKey::Profiler, input, weight, bias, M, N, eps);
}
std::tuple<Tensor,Tensor,Tensor> native_layer_norm_backward(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & rstd, const Tensor & weight, int64_t M, int64_t N, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_layer_norm_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, std::array<bool,3>)>();
  RECORD_FUNCTION("native_layer_norm_backward", std::vector<c10::IValue>({grad_out, input, mean, rstd, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_out, input, mean, rstd, weight, M, N, output_mask);
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
Tensor & neg_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::neg", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("neg_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor new_empty(const Tensor & self, IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::new_empty", "")
      .typed<Tensor (const Tensor &, IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("new_empty", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, self, size, options);
}
Tensor new_full(const Tensor & self, IntArrayRef size, Scalar fill_value, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::new_full", "")
      .typed<Tensor (const Tensor &, IntArrayRef, Scalar, const TensorOptions &)>();
  RECORD_FUNCTION("new_full", std::vector<c10::IValue>({self, fill_value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, Scalar, const TensorOptions &>(op, c10::DispatchKey::Profiler, self, size, fill_value, options);
}
Tensor new_zeros(const Tensor & self, IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::new_zeros", "")
      .typed<Tensor (const Tensor &, IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("new_zeros", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, self, size, options);
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
Tensor nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>();
  RECORD_FUNCTION("nll_loss2d_backward", std::vector<c10::IValue>({grad_output, self, target, weight, total_weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
Tensor & nll_loss2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>();
  RECORD_FUNCTION("nll_loss2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, weight, total_weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
std::tuple<Tensor,Tensor> nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d_forward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("nll_loss2d_forward", std::vector<c10::IValue>({self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, target, weight, reduction, ignore_index);
}
std::tuple<Tensor &,Tensor &> nll_loss2d_forward_out_output(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("nll_loss2d_forward_out", std::vector<c10::IValue>({output, total_weight, self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, output, total_weight, self, target, weight, reduction, ignore_index);
}
Tensor & nll_loss2d_out_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("nll_loss2d_out", std::vector<c10::IValue>({out, self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, weight, reduction, ignore_index);
}
Tensor nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>();
  RECORD_FUNCTION("nll_loss_backward", std::vector<c10::IValue>({grad_output, self, target, weight, total_weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
Tensor & nll_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>();
  RECORD_FUNCTION("nll_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, weight, total_weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
std::tuple<Tensor,Tensor> nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss_forward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("nll_loss_forward", std::vector<c10::IValue>({self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, target, weight, reduction, ignore_index);
}
std::tuple<Tensor &,Tensor &> nll_loss_forward_out_output(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("nll_loss_forward_out", std::vector<c10::IValue>({output, total_weight, self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, output, total_weight, self, target, weight, reduction, ignore_index);
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
Tensor norm_except_dim(const Tensor & v, int64_t pow, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm_except_dim", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("norm_except_dim", std::vector<c10::IValue>({v}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, v, pow, dim);
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
Tensor ones_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ones_like", "")
      .typed<Tensor (const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("ones_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, options, memory_format);
}
Tensor & ones_out_out(Tensor & out, IntArrayRef size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ones", "out")
      .typed<Tensor & (Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("ones_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, size);
}
Tensor orgqr(const Tensor & self, const Tensor & input2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::orgqr", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("orgqr", std::vector<c10::IValue>({self, input2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, input2);
}
Tensor & orgqr_out_out(Tensor & out, const Tensor & self, const Tensor & input2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::orgqr", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("orgqr_out", std::vector<c10::IValue>({out, self, input2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, input2);
}
Tensor ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ormqr", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("ormqr", std::vector<c10::IValue>({self, input2, input3}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, input2, input3, left, transpose);
}
Tensor & ormqr_out_out(Tensor & out, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ormqr", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("ormqr_out", std::vector<c10::IValue>({out, self, input2, input3}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, out, self, input2, input3, left, transpose);
}
int64_t output_nr(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::output_nr", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("output_nr", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor pairwise_distance(const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pairwise_distance", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, double, bool)>();
  RECORD_FUNCTION("pairwise_distance", std::vector<c10::IValue>({x1, x2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, double, bool>(op, c10::DispatchKey::Profiler, x1, x2, p, eps, keepdim);
}
Tensor pdist(const Tensor & self, double p) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pdist", "")
      .typed<Tensor (const Tensor &, double)>();
  RECORD_FUNCTION("pdist", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double>(op, c10::DispatchKey::Profiler, self, p);
}
Tensor permute(const Tensor & self, IntArrayRef dims) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::permute", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("permute", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, dims);
}
Tensor pin_memory(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pin_memory", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("pin_memory", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor pinverse(const Tensor & self, double rcond) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pinverse", "")
      .typed<Tensor (const Tensor &, double)>();
  RECORD_FUNCTION("pinverse", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double>(op, c10::DispatchKey::Profiler, self, rcond);
}
Tensor pixel_shuffle(const Tensor & self, int64_t upscale_factor) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pixel_shuffle", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("pixel_shuffle", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, upscale_factor);
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
Tensor & polygamma_out_out(Tensor & out, int64_t n, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::polygamma", "out")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("polygamma_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, out, n, self);
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
Tensor prelu(const Tensor & self, const Tensor & weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::prelu", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("prelu", std::vector<c10::IValue>({self, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, weight);
}
std::tuple<Tensor,Tensor> prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::prelu_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("prelu_backward", std::vector<c10::IValue>({grad_output, self, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, weight);
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
ScalarType promote_types(ScalarType type1, ScalarType type2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::promote_types", "")
      .typed<ScalarType (ScalarType, ScalarType)>();
  RECORD_FUNCTION("promote_types", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<ScalarType, ScalarType, ScalarType>(op, c10::DispatchKey::Profiler, type1, type2);
}
Tensor & put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::put_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("put_", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, index, source, accumulate);
}
int64_t q_per_channel_axis(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_per_channel_axis", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("q_per_channel_axis", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor q_per_channel_scales(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_per_channel_scales", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("q_per_channel_scales", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor q_per_channel_zero_points(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_per_channel_zero_points", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("q_per_channel_zero_points", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
double q_scale(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_scale", "")
      .typed<double (const Tensor &)>();
  RECORD_FUNCTION("q_scale", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<double, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
int64_t q_zero_point(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_zero_point", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("q_zero_point", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor> qr(const Tensor & self, bool some) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::qr", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool)>();
  RECORD_FUNCTION("qr", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, some);
}
std::tuple<Tensor &,Tensor &> qr_out_Q(Tensor & Q, Tensor & R, const Tensor & self, bool some) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::qr", "Q")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("qr_out", std::vector<c10::IValue>({Q, R, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, Q, R, self, some);
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
Tensor quantized_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & var, double eps, double output_scale, int64_t output_zero_point) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_batch_norm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, int64_t)>();
  RECORD_FUNCTION("quantized_batch_norm", std::vector<c10::IValue>({input, weight, bias, mean, var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, int64_t>(op, c10::DispatchKey::Profiler, input, weight, bias, mean, var, eps, output_scale, output_zero_point);
}
Tensor quantized_gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_gru_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("quantized_gru_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}
std::tuple<Tensor,Tensor> quantized_lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_lstm_cell", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("quantized_lstm_cell", std::vector<c10::IValue>({input, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}
Tensor quantized_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_max_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("quantized_max_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor quantized_rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_rnn_relu_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("quantized_rnn_relu_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}
Tensor quantized_rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_rnn_tanh_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("quantized_rnn_tanh_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
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
Tensor & rad2deg_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rad2deg", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("rad2deg_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
Tensor rand_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand_like", "")
      .typed<Tensor (const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("rand_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, options, memory_format);
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
Tensor randn_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn_like", "")
      .typed<Tensor (const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("randn_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, options, memory_format);
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
Tensor & range_out_out(Tensor & out, Scalar start, Scalar end, Scalar step) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::range", "out")
      .typed<Tensor & (Tensor &, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("range_out", std::vector<c10::IValue>({out, start, end, step}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, start, end, step);
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
Tensor & reciprocal_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reciprocal", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("reciprocal_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
Tensor reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad1d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad1d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, padding);
}
Tensor & reflection_pad1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad1d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad1d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, padding);
}
Tensor & reflection_pad1d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad1d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad1d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, padding);
}
Tensor reflection_pad2d(const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, padding);
}
Tensor reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad2d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, padding);
}
Tensor & reflection_pad2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
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
Tensor & renorm_out_out(Tensor & out, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::renorm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, int64_t, Scalar)>();
  RECORD_FUNCTION("renorm_out", std::vector<c10::IValue>({out, self, p, maxnorm}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, int64_t, Scalar>(op, c10::DispatchKey::Profiler, out, self, p, dim, maxnorm);
}
Tensor repeat(const Tensor & self, IntArrayRef repeats) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::repeat", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("repeat", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, repeats);
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
Tensor replication_pad1d(const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, padding);
}
Tensor replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad1d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad1d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, padding);
}
Tensor & replication_pad1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad1d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad1d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, padding);
}
Tensor & replication_pad1d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad1d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad1d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, padding);
}
Tensor replication_pad2d(const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, padding);
}
Tensor replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad2d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, padding);
}
Tensor & replication_pad2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, padding);
}
Tensor & replication_pad2d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, padding);
}
Tensor replication_pad3d(const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, padding);
}
Tensor replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad3d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, padding);
}
Tensor & replication_pad3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, padding);
}
Tensor & replication_pad3d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad3d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, padding);
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
Tensor reshape_as(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reshape_as", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("reshape_as", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & resize_(Tensor & self, IntArrayRef size, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::resize_", "")
      .typed<Tensor & (Tensor &, IntArrayRef, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("resize_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, size, memory_format);
}
Tensor & resize_as_(Tensor & self, const Tensor & the_template, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::resize_as_", "")
      .typed<Tensor & (Tensor &, const Tensor &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("resize_as_", std::vector<c10::IValue>({self, the_template}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, the_template, memory_format);
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
void retain_grad(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::retain_grad", "")
      .typed<void (const Tensor &)>();
  RECORD_FUNCTION("retain_grad", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<void, const Tensor &>(op, c10::DispatchKey::Profiler, self);
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
Tensor rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rnn_relu_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("rnn_relu_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh);
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
Tensor rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rnn_tanh_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("rnn_tanh_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh);
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
Tensor & round_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::round", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("round_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
Tensor rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, bool self_is_result) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu_with_noise_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, bool)>();
  RECORD_FUNCTION("rrelu_with_noise_backward", std::vector<c10::IValue>({grad_output, self, noise, lower, upper}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, bool>(op, c10::DispatchKey::Profiler, grad_output, self, noise, lower, upper, training, self_is_result);
}
Tensor & rrelu_with_noise_out_out(Tensor & out, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu_with_noise", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>)>();
  RECORD_FUNCTION("rrelu_with_noise_out", std::vector<c10::IValue>({out, self, noise, lower, upper}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, self, noise, lower, upper, training, generator);
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
Tensor & rsqrt_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rsqrt", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("rsqrt_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
Tensor scalar_tensor(Scalar s, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scalar_tensor", "")
      .typed<Tensor (Scalar, const TensorOptions &)>();
  RECORD_FUNCTION("scalar_tensor", std::vector<c10::IValue>({s}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Scalar, const TensorOptions &>(op, c10::DispatchKey::Profiler, s, options);
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
Tensor & searchsorted_out_Tensor_out(Tensor & out, const Tensor & sorted_sequence, const Tensor & self, bool out_int32, bool right) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::searchsorted", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("searchsorted_out", std::vector<c10::IValue>({out, sorted_sequence, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, out, sorted_sequence, self, out_int32, right);
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
Tensor sigmoid_backward(const Tensor & grad_output, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sigmoid_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("sigmoid_backward", std::vector<c10::IValue>({grad_output, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, output);
}
Tensor & sigmoid_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sigmoid_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("sigmoid_backward_out", std::vector<c10::IValue>({grad_input, grad_output, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output);
}
Tensor & sigmoid_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sigmoid", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("sigmoid_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
Tensor & sign_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sign", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("sign_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
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
std::tuple<Tensor,Tensor> slogdet(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slogdet", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &)>();
  RECORD_FUNCTION("slogdet", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor slow_conv3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv3d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv3d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor,Tensor,Tensor> slow_conv3d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv3d_backward", "output_mask")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>();
  RECORD_FUNCTION("slow_conv3d_backward", std::vector<c10::IValue>({grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv3d_backward_out_grad_input(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv3d_backward", "grad_input")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("slow_conv3d_backward_out", std::vector<c10::IValue>({grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> slow_conv3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv3d_forward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv3d_forward", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv3d_forward_out_output(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv3d_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv3d_forward_out", std::vector<c10::IValue>({output, finput, fgrad_input, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}
Tensor & slow_conv3d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv3d_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, weight, kernel_size, bias, stride, padding);
}
Tensor slow_conv_dilated2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_dilated2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv_dilated2d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding, dilation);
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
std::tuple<Tensor,Tensor,Tensor> slow_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_dilated3d_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,3>)>();
  RECORD_FUNCTION("slow_conv_dilated3d_backward", std::vector<c10::IValue>({grad_output, self, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
Tensor slow_conv_transpose2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv_transpose2d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor,Tensor,Tensor> slow_conv_transpose2d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose2d_backward", "output_mask")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>();
  RECORD_FUNCTION("slow_conv_transpose2d_backward", std::vector<c10::IValue>({grad_output, self, weight, columns, ones}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv_transpose2d_backward_out_grad_output(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose2d_backward", "grad_output")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("slow_conv_transpose2d_backward_out", std::vector<c10::IValue>({grad_input, grad_weight, grad_bias, grad_output, self, weight, columns, ones}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones);
}
Tensor & slow_conv_transpose2d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv_transpose2d_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
Tensor slow_conv_transpose3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose3d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv_transpose3d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor,Tensor,Tensor> slow_conv_transpose3d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose3d_backward", "output_mask")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>();
  RECORD_FUNCTION("slow_conv_transpose3d_backward", std::vector<c10::IValue>({grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv_transpose3d_backward_out_grad_output(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose3d_backward", "grad_output")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("slow_conv_transpose3d_backward_out", std::vector<c10::IValue>({grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input);
}
Tensor & slow_conv_transpose3d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv_transpose3d_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
Tensor smm(const Tensor & self, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::smm", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("smm", std::vector<c10::IValue>({self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, mat2);
}
Tensor smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::smooth_l1_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("smooth_l1_loss", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, reduction);
}
Tensor smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::smooth_l1_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("smooth_l1_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, reduction);
}
Tensor & smooth_l1_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::smooth_l1_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("smooth_l1_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, reduction);
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
Tensor soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::soft_margin_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("soft_margin_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, reduction);
}
Tensor & soft_margin_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::soft_margin_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("soft_margin_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, reduction);
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
Tensor softplus(const Tensor & self, Scalar beta, Scalar threshold) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softplus", "")
      .typed<Tensor (const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("softplus", std::vector<c10::IValue>({self, beta, threshold}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, beta, threshold);
}
Tensor softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softplus_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>();
  RECORD_FUNCTION("softplus_backward", std::vector<c10::IValue>({grad_output, self, beta, threshold, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, beta, threshold, output);
}
Tensor & softplus_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softplus_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>();
  RECORD_FUNCTION("softplus_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, beta, threshold, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, beta, threshold, output);
}
Tensor & softplus_out_out(Tensor & out, const Tensor & self, Scalar beta, Scalar threshold) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softplus", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("softplus_out", std::vector<c10::IValue>({out, self, beta, threshold}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, beta, threshold);
}
Tensor softshrink(const Tensor & self, Scalar lambd) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softshrink", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("softshrink", std::vector<c10::IValue>({self, lambd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, lambd);
}
Tensor softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softshrink_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("softshrink_backward", std::vector<c10::IValue>({grad_output, self, lambd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, grad_output, self, lambd);
}
Tensor & softshrink_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softshrink_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("softshrink_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, lambd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, lambd);
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
std::tuple<Tensor &,Tensor &> solve_out_solution(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::solve", "solution")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("solve_out", std::vector<c10::IValue>({solution, lu, self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, solution, lu, self, A);
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
int64_t sparse_dim(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sparse_dim", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("sparse_dim", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
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
Tensor & sparse_resize_and_clear_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sparse_resize_and_clear_", "")
      .typed<Tensor & (Tensor &, IntArrayRef, int64_t, int64_t)>();
  RECORD_FUNCTION("sparse_resize_and_clear_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, size, sparse_dim, dense_dim);
}
std::vector<Tensor> split_Tensor(const Tensor & self, int64_t split_size, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::split", "Tensor")
      .typed<std::vector<Tensor> (const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("split", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, split_size, dim);
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
Tensor sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sspaddmm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("sspaddmm", std::vector<c10::IValue>({self, mat1, mat2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, mat1, mat2, beta, alpha);
}
Tensor & sspaddmm_out_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sspaddmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("sspaddmm_out", std::vector<c10::IValue>({out, self, mat1, mat2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, mat1, mat2, beta, alpha);
}
Tensor stack(TensorList tensors, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::stack", "")
      .typed<Tensor (TensorList, int64_t)>();
  RECORD_FUNCTION("stack", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList, int64_t>(op, c10::DispatchKey::Profiler, tensors, dim);
}
Tensor & stack_out_out(Tensor & out, TensorList tensors, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::stack", "out")
      .typed<Tensor & (Tensor &, TensorList, int64_t)>();
  RECORD_FUNCTION("stack_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, TensorList, int64_t>(op, c10::DispatchKey::Profiler, out, tensors, dim);
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
Tensor stft(const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::stft", "")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("stft", std::vector<c10::IValue>({self, window}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, n_fft, hop_length, win_length, window, normalized, onesided);
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
Tensor & sub_out_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sub", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("sub_out", std::vector<c10::IValue>({out, self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other, alpha);
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
Tensor sum_to_size(const Tensor & self, IntArrayRef size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sum_to_size", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("sum_to_size", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, size);
}
std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some, bool compute_uv) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::svd", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool)>();
  RECORD_FUNCTION("svd", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, some, compute_uv);
}
std::tuple<Tensor &,Tensor &,Tensor &> svd_out_U(Tensor & U, Tensor & S, Tensor & V, const Tensor & self, bool some, bool compute_uv) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::svd", "U")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("svd_out", std::vector<c10::IValue>({U, S, V, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, U, S, V, self, some, compute_uv);
}
std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::symeig", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool, bool)>();
  RECORD_FUNCTION("symeig", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, eigenvectors, upper);
}
std::tuple<Tensor &,Tensor &> symeig_out_e(Tensor & e, Tensor & V, const Tensor & self, bool eigenvectors, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::symeig", "e")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("symeig_out", std::vector<c10::IValue>({e, V, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, e, V, self, eigenvectors, upper);
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
Tensor & take_out_out(Tensor & out, const Tensor & self, const Tensor & index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::take", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("take_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, index);
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
Tensor tanh_backward(const Tensor & grad_output, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tanh_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("tanh_backward", std::vector<c10::IValue>({grad_output, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, output);
}
Tensor & tanh_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tanh_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("tanh_backward_out", std::vector<c10::IValue>({grad_input, grad_output, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output);
}
Tensor & tanh_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tanh", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("tanh_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor tensordot(const Tensor & self, const Tensor & other, IntArrayRef dims_self, IntArrayRef dims_other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tensordot", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("tensordot", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, other, dims_self, dims_other);
}
Tensor thnn_conv2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv2d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d_backward", "output_mask")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>();
  RECORD_FUNCTION("thnn_conv2d_backward", std::vector<c10::IValue>({grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_backward_out_grad_input(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d_backward", "grad_input")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("thnn_conv2d_backward_out", std::vector<c10::IValue>({grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d_forward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv2d_forward", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_forward_out_output(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv2d_forward_out", std::vector<c10::IValue>({output, finput, fgrad_input, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}
Tensor & thnn_conv2d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv2d_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, weight, kernel_size, bias, stride, padding);
}
Tensor thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv_depthwise2d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor,Tensor> thnn_conv_depthwise2d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,2> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d_backward", "output_mask")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,2>)>();
  RECORD_FUNCTION("thnn_conv_depthwise2d_backward", std::vector<c10::IValue>({grad_output, self, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,2>>(op, c10::DispatchKey::Profiler, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
std::tuple<Tensor &,Tensor &> thnn_conv_depthwise2d_backward_out_grad_input(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d_backward", "grad_input")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv_depthwise2d_backward_out", std::vector<c10::IValue>({grad_input, grad_weight, grad_output, self, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_weight, grad_output, self, weight, kernel_size, stride, padding, dilation);
}
Tensor thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d_forward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv_depthwise2d_forward", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding, dilation);
}
Tensor & thnn_conv_depthwise2d_forward_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d_forward", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv_depthwise2d_forward_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, weight, kernel_size, bias, stride, padding, dilation);
}
Tensor & thnn_conv_depthwise2d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv_depthwise2d_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, weight, kernel_size, bias, stride, padding, dilation);
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
Tensor threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::threshold_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("threshold_backward", std::vector<c10::IValue>({grad_output, self, threshold}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, grad_output, self, threshold);
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
Tensor to_dense(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_dense", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("to_dense", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor to_dense_backward(const Tensor & grad, const Tensor & input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_dense_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("to_dense_backward", std::vector<c10::IValue>({grad, input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad, input);
}
Tensor to_mkldnn(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_mkldnn", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("to_mkldnn", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor to_mkldnn_backward(const Tensor & grad, const Tensor & input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_mkldnn_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("to_mkldnn_backward", std::vector<c10::IValue>({grad, input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad, input);
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
std::tuple<Tensor &,Tensor &> topk_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::topk", "values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, int64_t, bool, bool)>();
  RECORD_FUNCTION("topk_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, values, indices, self, k, dim, largest, sorted);
}
Tensor trace(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::trace", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("trace", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
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
std::tuple<Tensor,Tensor> triangular_solve(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::triangular_solve", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool, bool, bool)>();
  RECORD_FUNCTION("triangular_solve", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool, bool, bool>(op, c10::DispatchKey::Profiler, self, A, upper, transpose, unitriangular);
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
Tensor & tril_out_out(Tensor & out, const Tensor & self, int64_t diagonal) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tril", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("tril_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, diagonal);
}
Tensor triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::triplet_margin_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, double, double, double, bool, int64_t)>();
  RECORD_FUNCTION("triplet_margin_loss", std::vector<c10::IValue>({anchor, positive, negative}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, double, double, double, bool, int64_t>(op, c10::DispatchKey::Profiler, anchor, positive, negative, margin, p, eps, swap, reduction);
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
Tensor & triu_out_out(Tensor & out, const Tensor & self, int64_t diagonal) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::triu", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("triu_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, diagonal);
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
Tensor & trunc_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::trunc", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("trunc_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor type_as(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::type_as", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("type_as", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
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
Tensor unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unfold", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("unfold", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, dimension, size, step);
}
Tensor unfold_backward(const Tensor & grad_in, IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unfold_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("unfold_backward", std::vector<c10::IValue>({grad_in}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, grad_in, input_sizes, dim, size, step);
}
Tensor & uniform_(Tensor & self, double from, double to, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::uniform_", "")
      .typed<Tensor & (Tensor &, double, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("uniform_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, from, to, generator);
}
std::tuple<Tensor,Tensor,Tensor> unique_consecutive(const Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unique_consecutive", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("unique_consecutive", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, return_inverse, return_counts, dim);
}
std::tuple<Tensor,Tensor,Tensor> unique_dim(const Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unique_dim", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, int64_t, bool, bool, bool)>();
  RECORD_FUNCTION("unique_dim", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, int64_t, bool, bool, bool>(op, c10::DispatchKey::Profiler, self, dim, sorted, return_inverse, return_counts);
}
std::tuple<Tensor,Tensor,Tensor> unique_dim_consecutive(const Tensor & self, int64_t dim, bool return_inverse, bool return_counts) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unique_dim_consecutive", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, int64_t, bool, bool)>();
  RECORD_FUNCTION("unique_dim_consecutive", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, dim, return_inverse, return_counts);
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
Tensor upsample_bicubic2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bicubic2d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bicubic2d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}
Tensor & upsample_bicubic2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bicubic2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bicubic2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}
Tensor & upsample_bicubic2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bicubic2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bicubic2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, out, self, output_size, align_corners, scales_h, scales_w);
}
Tensor upsample_bilinear2d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bilinear2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bilinear2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, self, output_size, align_corners, scales_h, scales_w);
}
Tensor upsample_bilinear2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bilinear2d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bilinear2d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}
Tensor & upsample_bilinear2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bilinear2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bilinear2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}
Tensor & upsample_bilinear2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bilinear2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bilinear2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, out, self, output_size, align_corners, scales_h, scales_w);
}
Tensor upsample_linear1d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_linear1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_linear1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<double>>(op, c10::DispatchKey::Profiler, self, output_size, align_corners, scales);
}
Tensor upsample_linear1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_linear1d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_linear1d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, align_corners, scales);
}
Tensor & upsample_linear1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_linear1d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_linear1d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, align_corners, scales);
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
Tensor upsample_nearest1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest1d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest1d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, scales);
}
Tensor & upsample_nearest1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest1d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest1d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, scales);
}
Tensor & upsample_nearest1d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest1d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest1d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, c10::optional<double>>(op, c10::DispatchKey::Profiler, out, self, output_size, scales);
}
Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, self, output_size, scales_h, scales_w);
}
Tensor upsample_nearest2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest2d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest2d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, scales_h, scales_w);
}
Tensor & upsample_nearest2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, scales_h, scales_w);
}
Tensor & upsample_nearest2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, out, self, output_size, scales_h, scales_w);
}
Tensor upsample_nearest3d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, self, output_size, scales_d, scales_h, scales_w);
}
Tensor upsample_nearest3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest3d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest3d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
}
Tensor & upsample_nearest3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
}
Tensor & upsample_nearest3d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest3d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, out, self, output_size, scales_d, scales_h, scales_w);
}
Tensor upsample_trilinear3d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_trilinear3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_trilinear3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, self, output_size, align_corners, scales_d, scales_h, scales_w);
}
Tensor upsample_trilinear3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_trilinear3d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_trilinear3d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
}
Tensor & upsample_trilinear3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_trilinear3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_trilinear3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
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
Tensor vander(const Tensor & x, c10::optional<int64_t> N, bool increasing) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::vander", "")
      .typed<Tensor (const Tensor &, c10::optional<int64_t>, bool)>();
  RECORD_FUNCTION("vander", std::vector<c10::IValue>({x}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<int64_t>, bool>(op, c10::DispatchKey::Profiler, x, N, increasing);
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
Tensor zeros_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::zeros_like", "")
      .typed<Tensor (const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("zeros_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, options, memory_format);
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
  m.impl_UNBOXED("__ilshift__.Scalar", &ProfiledType::__ilshift___Scalar);
  m.impl_UNBOXED("__ilshift__.Tensor", &ProfiledType::__ilshift___Tensor);
  m.impl_UNBOXED("__ior__.Scalar", &ProfiledType::__ior___Scalar);
  m.impl_UNBOXED("__ior__.Tensor", &ProfiledType::__ior___Tensor);
  m.impl_UNBOXED("__irshift__.Scalar", &ProfiledType::__irshift___Scalar);
  m.impl_UNBOXED("__irshift__.Tensor", &ProfiledType::__irshift___Tensor);
  m.impl_UNBOXED("__ixor__.Scalar", &ProfiledType::__ixor___Scalar);
  m.impl_UNBOXED("__ixor__.Tensor", &ProfiledType::__ixor___Tensor);
  m.impl("__lshift__.Scalar", TORCH_FN(ProfiledType::__lshift___Scalar));
  m.impl("__lshift__.Tensor", TORCH_FN(ProfiledType::__lshift___Tensor));
  m.impl("__or__.Scalar", TORCH_FN(ProfiledType::__or___Scalar));
  m.impl("__or__.Tensor", TORCH_FN(ProfiledType::__or___Tensor));
  m.impl("__rshift__.Scalar", TORCH_FN(ProfiledType::__rshift___Scalar));
  m.impl("__rshift__.Tensor", TORCH_FN(ProfiledType::__rshift___Tensor));
  m.impl("__xor__.Scalar", TORCH_FN(ProfiledType::__xor___Scalar));
  m.impl("__xor__.Tensor", TORCH_FN(ProfiledType::__xor___Tensor));
  m.impl("_adaptive_avg_pool2d", TORCH_FN(ProfiledType::_adaptive_avg_pool2d));
  m.impl("_adaptive_avg_pool2d_backward", TORCH_FN(ProfiledType::_adaptive_avg_pool2d_backward));
  m.impl_UNBOXED("_addmv_impl_", &ProfiledType::_addmv_impl_);
  m.impl("_addr", TORCH_FN(ProfiledType::_addr));
  m.impl_UNBOXED("_addr_", &ProfiledType::_addr_);
  m.impl_UNBOXED("_addr.out", &ProfiledType::_addr_out_out);
  m.impl_UNBOXED("_amp_non_finite_check_and_unscale_", &ProfiledType::_amp_non_finite_check_and_unscale_);
  m.impl_UNBOXED("_amp_update_scale", &ProfiledType::_amp_update_scale);
  m.impl_UNBOXED("_baddbmm_mkl_", &ProfiledType::_baddbmm_mkl_);
  m.impl_UNBOXED("_batch_norm_impl_index", &ProfiledType::_batch_norm_impl_index);
  m.impl_UNBOXED("_batch_norm_impl_index_backward", &ProfiledType::_batch_norm_impl_index_backward);
  m.impl("_bmm", TORCH_FN(ProfiledType::_bmm));
  m.impl_UNBOXED("_bmm.out", &ProfiledType::_bmm_out_out);
  m.impl("_cast_Byte", TORCH_FN(ProfiledType::_cast_Byte));
  m.impl("_cast_Char", TORCH_FN(ProfiledType::_cast_Char));
  m.impl("_cast_Double", TORCH_FN(ProfiledType::_cast_Double));
  m.impl("_cast_Float", TORCH_FN(ProfiledType::_cast_Float));
  m.impl("_cast_Half", TORCH_FN(ProfiledType::_cast_Half));
  m.impl("_cast_Int", TORCH_FN(ProfiledType::_cast_Int));
  m.impl("_cast_Long", TORCH_FN(ProfiledType::_cast_Long));
  m.impl("_cast_Short", TORCH_FN(ProfiledType::_cast_Short));
  m.impl("_cat", TORCH_FN(ProfiledType::_cat));
  m.impl_UNBOXED("_cat.out", &ProfiledType::_cat_out_out);
  m.impl("_cdist_backward", TORCH_FN(ProfiledType::_cdist_backward));
  m.impl("_cdist_forward", TORCH_FN(ProfiledType::_cdist_forward));
  m.impl("_cholesky_helper", TORCH_FN(ProfiledType::_cholesky_helper));
  m.impl("_cholesky_solve_helper", TORCH_FN(ProfiledType::_cholesky_solve_helper));
  m.impl("_choose_qparams_per_tensor", TORCH_FN(ProfiledType::_choose_qparams_per_tensor));
  m.impl_UNBOXED("_coalesced_", &ProfiledType::_coalesced_);
  m.impl_UNBOXED("_convolution", &ProfiledType::_convolution);
  m.impl_UNBOXED("_convolution_double_backward", &ProfiledType::_convolution_double_backward);
  m.impl_UNBOXED("_convolution_nogroup", &ProfiledType::_convolution_nogroup);
  m.impl("_copy_from", TORCH_FN(ProfiledType::_copy_from));
  m.impl("_ctc_loss", TORCH_FN(ProfiledType::_ctc_loss));
  m.impl("_ctc_loss_backward", TORCH_FN(ProfiledType::_ctc_loss_backward));
  m.impl("_cudnn_ctc_loss", TORCH_FN(ProfiledType::_cudnn_ctc_loss));
  m.impl_UNBOXED("_cudnn_init_dropout_state", &ProfiledType::_cudnn_init_dropout_state);
  m.impl_UNBOXED("_cudnn_rnn", &ProfiledType::_cudnn_rnn);
  m.impl_UNBOXED("_cudnn_rnn_backward", &ProfiledType::_cudnn_rnn_backward);
  m.impl("_cudnn_rnn_flatten_weight", TORCH_FN(ProfiledType::_cudnn_rnn_flatten_weight));
  m.impl("_cufft_clear_plan_cache", TORCH_FN(ProfiledType::_cufft_clear_plan_cache));
  m.impl("_cufft_get_plan_cache_max_size", TORCH_FN(ProfiledType::_cufft_get_plan_cache_max_size));
  m.impl("_cufft_get_plan_cache_size", TORCH_FN(ProfiledType::_cufft_get_plan_cache_size));
  m.impl("_cufft_set_plan_cache_max_size", TORCH_FN(ProfiledType::_cufft_set_plan_cache_max_size));
  m.impl_UNBOXED("_cummax_helper", &ProfiledType::_cummax_helper);
  m.impl_UNBOXED("_cummin_helper", &ProfiledType::_cummin_helper);
  m.impl("_cumprod", TORCH_FN(ProfiledType::_cumprod));
  m.impl_UNBOXED("_cumprod.out", &ProfiledType::_cumprod_out_out);
  m.impl("_cumsum", TORCH_FN(ProfiledType::_cumsum));
  m.impl_UNBOXED("_cumsum.out", &ProfiledType::_cumsum_out_out);
  m.impl("_debug_has_internal_overlap", TORCH_FN(ProfiledType::_debug_has_internal_overlap));
  m.impl("_dimI", TORCH_FN(ProfiledType::_dimI));
  m.impl("_dimV", TORCH_FN(ProfiledType::_dimV));
  m.impl("_dim_arange", TORCH_FN(ProfiledType::_dim_arange));
  m.impl("_dirichlet_grad", TORCH_FN(ProfiledType::_dirichlet_grad));
  m.impl_UNBOXED("_embedding_bag", &ProfiledType::_embedding_bag);
  m.impl_UNBOXED("_embedding_bag_backward", &ProfiledType::_embedding_bag_backward);
  m.impl_UNBOXED("_embedding_bag_dense_backward", &ProfiledType::_embedding_bag_dense_backward);
  m.impl("_embedding_bag_per_sample_weights_backward", TORCH_FN(ProfiledType::_embedding_bag_per_sample_weights_backward));
  m.impl_UNBOXED("_embedding_bag_sparse_backward", &ProfiledType::_embedding_bag_sparse_backward);
  m.impl_UNBOXED("_empty_affine_quantized", &ProfiledType::_empty_affine_quantized);
  m.impl_UNBOXED("_empty_per_channel_affine_quantized", &ProfiledType::_empty_per_channel_affine_quantized);
  m.impl("_euclidean_dist", TORCH_FN(ProfiledType::_euclidean_dist));
  m.impl("_fft_with_size", TORCH_FN(ProfiledType::_fft_with_size));
  m.impl_UNBOXED("_fused_dropout", &ProfiledType::_fused_dropout);
  m.impl("_gather_sparse_backward", TORCH_FN(ProfiledType::_gather_sparse_backward));
  m.impl("_has_compatible_shallow_copy_type", TORCH_FN(ProfiledType::_has_compatible_shallow_copy_type));
  m.impl_UNBOXED("_index_copy_", &ProfiledType::_index_copy_);
  m.impl_UNBOXED("_index_put_impl_", &ProfiledType::_index_put_impl_);
  m.impl("_indices", TORCH_FN(ProfiledType::_indices));
  m.impl("_inverse_helper", TORCH_FN(ProfiledType::_inverse_helper));
  m.impl("_local_scalar_dense", TORCH_FN(ProfiledType::_local_scalar_dense));
  m.impl("_log_softmax", TORCH_FN(ProfiledType::_log_softmax));
  m.impl("_log_softmax_backward_data", TORCH_FN(ProfiledType::_log_softmax_backward_data));
  m.impl("_logcumsumexp", TORCH_FN(ProfiledType::_logcumsumexp));
  m.impl_UNBOXED("_logcumsumexp.out", &ProfiledType::_logcumsumexp_out_out);
  m.impl("_lu_solve_helper", TORCH_FN(ProfiledType::_lu_solve_helper));
  m.impl("_lu_with_info", TORCH_FN(ProfiledType::_lu_with_info));
  m.impl("_make_per_channel_quantized_tensor", TORCH_FN(ProfiledType::_make_per_channel_quantized_tensor));
  m.impl("_make_per_tensor_quantized_tensor", TORCH_FN(ProfiledType::_make_per_tensor_quantized_tensor));
  m.impl("_masked_scale", TORCH_FN(ProfiledType::_masked_scale));
  m.impl("_mkldnn_reshape", TORCH_FN(ProfiledType::_mkldnn_reshape));
  m.impl("_mkldnn_transpose", TORCH_FN(ProfiledType::_mkldnn_transpose));
  m.impl_UNBOXED("_mkldnn_transpose_", &ProfiledType::_mkldnn_transpose_);
  m.impl("_mode", TORCH_FN(ProfiledType::_mode));
  m.impl_UNBOXED("_mode.values", &ProfiledType::_mode_out_values);
  m.impl_UNBOXED("_multinomial_alias_draw", &ProfiledType::_multinomial_alias_draw);
  m.impl("_multinomial_alias_setup", TORCH_FN(ProfiledType::_multinomial_alias_setup));
  m.impl("_nnpack_available", TORCH_FN(ProfiledType::_nnpack_available));
  m.impl_UNBOXED("_nnpack_spatial_convolution", &ProfiledType::_nnpack_spatial_convolution);
  m.impl("_nnpack_spatial_convolution_backward", TORCH_FN(ProfiledType::_nnpack_spatial_convolution_backward));
  m.impl("_nnpack_spatial_convolution_backward_input", TORCH_FN(ProfiledType::_nnpack_spatial_convolution_backward_input));
  m.impl("_nnpack_spatial_convolution_backward_weight", TORCH_FN(ProfiledType::_nnpack_spatial_convolution_backward_weight));
  m.impl("_nnz", TORCH_FN(ProfiledType::_nnz));
  m.impl("_pack_padded_sequence", TORCH_FN(ProfiledType::_pack_padded_sequence));
  m.impl("_pack_padded_sequence_backward", TORCH_FN(ProfiledType::_pack_padded_sequence_backward));
  m.impl("_pad_packed_sequence", TORCH_FN(ProfiledType::_pad_packed_sequence));
  m.impl("_pdist_backward", TORCH_FN(ProfiledType::_pdist_backward));
  m.impl("_pdist_forward", TORCH_FN(ProfiledType::_pdist_forward));
  m.impl("_qr_helper", TORCH_FN(ProfiledType::_qr_helper));
  m.impl("_reshape_from_tensor", TORCH_FN(ProfiledType::_reshape_from_tensor));
  m.impl("_s_where", TORCH_FN(ProfiledType::_s_where));
  m.impl_UNBOXED("_sample_dirichlet", &ProfiledType::_sample_dirichlet);
  m.impl("_shape_as_tensor", TORCH_FN(ProfiledType::_shape_as_tensor));
  m.impl_UNBOXED("_sobol_engine_draw", &ProfiledType::_sobol_engine_draw);
  m.impl_UNBOXED("_sobol_engine_ff_", &ProfiledType::_sobol_engine_ff_);
  m.impl_UNBOXED("_sobol_engine_initialize_state_", &ProfiledType::_sobol_engine_initialize_state_);
  m.impl_UNBOXED("_sobol_engine_scramble_", &ProfiledType::_sobol_engine_scramble_);
  m.impl("_softmax", TORCH_FN(ProfiledType::_softmax));
  m.impl("_softmax_backward_data", TORCH_FN(ProfiledType::_softmax_backward_data));
  m.impl("_solve_helper", TORCH_FN(ProfiledType::_solve_helper));
  m.impl("_sparse_addmm", TORCH_FN(ProfiledType::_sparse_addmm));
  m.impl_UNBOXED("_sparse_coo_tensor_unsafe", &ProfiledType::_sparse_coo_tensor_unsafe);
  m.impl_UNBOXED("_sparse_coo_tensor_with_dims", &ProfiledType::_sparse_coo_tensor_with_dims);
  m.impl_UNBOXED("_sparse_coo_tensor_with_dims_and_tensors", &ProfiledType::_sparse_coo_tensor_with_dims_and_tensors);
  m.impl_UNBOXED("_sparse_log_softmax.int", &ProfiledType::_sparse_log_softmax_int);
  m.impl_UNBOXED("_sparse_log_softmax.Dimname", &ProfiledType::_sparse_log_softmax_Dimname);
  m.impl("_sparse_log_softmax", TORCH_FN(ProfiledType::_sparse_log_softmax));
  m.impl_UNBOXED("_sparse_log_softmax_backward_data", &ProfiledType::_sparse_log_softmax_backward_data);
  m.impl("_sparse_mm", TORCH_FN(ProfiledType::_sparse_mm));
  m.impl_UNBOXED("_sparse_softmax.int", &ProfiledType::_sparse_softmax_int);
  m.impl_UNBOXED("_sparse_softmax.Dimname", &ProfiledType::_sparse_softmax_Dimname);
  m.impl("_sparse_softmax", TORCH_FN(ProfiledType::_sparse_softmax));
  m.impl_UNBOXED("_sparse_softmax_backward_data", &ProfiledType::_sparse_softmax_backward_data);
  m.impl("_sparse_sum", TORCH_FN(ProfiledType::_sparse_sum));
  m.impl_UNBOXED("_sparse_sum.dtype", &ProfiledType::_sparse_sum_dtype);
  m.impl("_sparse_sum.dim", TORCH_FN(ProfiledType::_sparse_sum_dim));
  m.impl_UNBOXED("_sparse_sum.dim_dtype", &ProfiledType::_sparse_sum_dim_dtype);
  m.impl("_sparse_sum_backward", TORCH_FN(ProfiledType::_sparse_sum_backward));
  m.impl_UNBOXED("_standard_gamma", &ProfiledType::_standard_gamma);
  m.impl("_standard_gamma_grad", TORCH_FN(ProfiledType::_standard_gamma_grad));
  m.impl("_svd_helper", TORCH_FN(ProfiledType::_svd_helper));
  m.impl("_symeig_helper", TORCH_FN(ProfiledType::_symeig_helper));
  m.impl("_test_serialization_subcmul", TORCH_FN(ProfiledType::_test_serialization_subcmul));
  m.impl_UNBOXED("_thnn_differentiable_gru_cell_backward", &ProfiledType::_thnn_differentiable_gru_cell_backward);
  m.impl_UNBOXED("_thnn_differentiable_lstm_cell_backward", &ProfiledType::_thnn_differentiable_lstm_cell_backward);
  m.impl_UNBOXED("_thnn_fused_gru_cell", &ProfiledType::_thnn_fused_gru_cell);
  m.impl("_thnn_fused_gru_cell_backward", TORCH_FN(ProfiledType::_thnn_fused_gru_cell_backward));
  m.impl_UNBOXED("_thnn_fused_lstm_cell", &ProfiledType::_thnn_fused_lstm_cell);
  m.impl_UNBOXED("_thnn_fused_lstm_cell_backward", &ProfiledType::_thnn_fused_lstm_cell_backward);
  m.impl("_triangular_solve_helper", TORCH_FN(ProfiledType::_triangular_solve_helper));
  m.impl("_trilinear", TORCH_FN(ProfiledType::_trilinear));
  m.impl("_unique", TORCH_FN(ProfiledType::_unique));
  m.impl("_unique2", TORCH_FN(ProfiledType::_unique2));
  m.impl("_unsafe_view", TORCH_FN(ProfiledType::_unsafe_view));
  m.impl("_use_cudnn_ctc_loss", TORCH_FN(ProfiledType::_use_cudnn_ctc_loss));
  m.impl("_use_cudnn_rnn_flatten_weight", TORCH_FN(ProfiledType::_use_cudnn_rnn_flatten_weight));
  m.impl("_values", TORCH_FN(ProfiledType::_values));
  m.impl("_version", TORCH_FN(ProfiledType::_version));
  m.impl("_weight_norm", TORCH_FN(ProfiledType::_weight_norm));
  m.impl("_weight_norm_cuda_interface", TORCH_FN(ProfiledType::_weight_norm_cuda_interface));
  m.impl("_weight_norm_cuda_interface_backward", TORCH_FN(ProfiledType::_weight_norm_cuda_interface_backward));
  m.impl("_weight_norm_differentiable_backward", TORCH_FN(ProfiledType::_weight_norm_differentiable_backward));
  m.impl("abs", TORCH_FN(ProfiledType::abs));
  m.impl_UNBOXED("abs_", &ProfiledType::abs_);
  m.impl_UNBOXED("abs.out", &ProfiledType::abs_out_out);
  m.impl("absolute", TORCH_FN(ProfiledType::absolute));
  m.impl_UNBOXED("absolute_", &ProfiledType::absolute_);
  m.impl_UNBOXED("absolute.out", &ProfiledType::absolute_out_out);
  m.impl("acos", TORCH_FN(ProfiledType::acos));
  m.impl_UNBOXED("acos_", &ProfiledType::acos_);
  m.impl_UNBOXED("acos.out", &ProfiledType::acos_out_out);
  m.impl("acosh", TORCH_FN(ProfiledType::acosh));
  m.impl_UNBOXED("acosh_", &ProfiledType::acosh_);
  m.impl_UNBOXED("acosh.out", &ProfiledType::acosh_out_out);
  m.impl("adaptive_avg_pool1d", TORCH_FN(ProfiledType::adaptive_avg_pool1d));
  m.impl("adaptive_avg_pool2d", TORCH_FN(ProfiledType::adaptive_avg_pool2d));
  m.impl_UNBOXED("adaptive_avg_pool2d.out", &ProfiledType::adaptive_avg_pool2d_out_out);
  m.impl("adaptive_avg_pool3d", TORCH_FN(ProfiledType::adaptive_avg_pool3d));
  m.impl("adaptive_avg_pool3d_backward", TORCH_FN(ProfiledType::adaptive_avg_pool3d_backward));
  m.impl_UNBOXED("adaptive_avg_pool3d_backward.grad_input", &ProfiledType::adaptive_avg_pool3d_backward_out_grad_input);
  m.impl_UNBOXED("adaptive_avg_pool3d.out", &ProfiledType::adaptive_avg_pool3d_out_out);
  m.impl("adaptive_max_pool1d", TORCH_FN(ProfiledType::adaptive_max_pool1d));
  m.impl("adaptive_max_pool2d", TORCH_FN(ProfiledType::adaptive_max_pool2d));
  m.impl("adaptive_max_pool2d_backward", TORCH_FN(ProfiledType::adaptive_max_pool2d_backward));
  m.impl_UNBOXED("adaptive_max_pool2d_backward.grad_input", &ProfiledType::adaptive_max_pool2d_backward_out_grad_input);
  m.impl_UNBOXED("adaptive_max_pool2d.out", &ProfiledType::adaptive_max_pool2d_out_out);
  m.impl("adaptive_max_pool3d", TORCH_FN(ProfiledType::adaptive_max_pool3d));
  m.impl("adaptive_max_pool3d_backward", TORCH_FN(ProfiledType::adaptive_max_pool3d_backward));
  m.impl_UNBOXED("adaptive_max_pool3d_backward.grad_input", &ProfiledType::adaptive_max_pool3d_backward_out_grad_input);
  m.impl_UNBOXED("adaptive_max_pool3d.out", &ProfiledType::adaptive_max_pool3d_out_out);
  m.impl("add.Tensor", TORCH_FN(ProfiledType::add_Tensor));
  m.impl("add.Scalar", TORCH_FN(ProfiledType::add_Scalar));
  m.impl_UNBOXED("add_.Tensor", &ProfiledType::add__Tensor);
  m.impl_UNBOXED("add_.Scalar", &ProfiledType::add__Scalar);
  m.impl_UNBOXED("add.out", &ProfiledType::add_out_out);
  m.impl("addbmm", TORCH_FN(ProfiledType::addbmm));
  m.impl_UNBOXED("addbmm_", &ProfiledType::addbmm_);
  m.impl_UNBOXED("addbmm.out", &ProfiledType::addbmm_out_out);
  m.impl("addcdiv", TORCH_FN(ProfiledType::addcdiv));
  m.impl_UNBOXED("addcdiv_", &ProfiledType::addcdiv_);
  m.impl_UNBOXED("addcdiv.out", &ProfiledType::addcdiv_out_out);
  m.impl("addcmul", TORCH_FN(ProfiledType::addcmul));
  m.impl_UNBOXED("addcmul_", &ProfiledType::addcmul_);
  m.impl_UNBOXED("addcmul.out", &ProfiledType::addcmul_out_out);
  m.impl("addmm", TORCH_FN(ProfiledType::addmm));
  m.impl_UNBOXED("addmm_", &ProfiledType::addmm_);
  m.impl_UNBOXED("addmm.out", &ProfiledType::addmm_out_out);
  m.impl("addmv", TORCH_FN(ProfiledType::addmv));
  m.impl_UNBOXED("addmv_", &ProfiledType::addmv_);
  m.impl_UNBOXED("addmv.out", &ProfiledType::addmv_out_out);
  m.impl("addr", TORCH_FN(ProfiledType::addr));
  m.impl_UNBOXED("addr_", &ProfiledType::addr_);
  m.impl_UNBOXED("addr.out", &ProfiledType::addr_out_out);
  m.impl("affine_grid_generator", TORCH_FN(ProfiledType::affine_grid_generator));
  m.impl("affine_grid_generator_backward", TORCH_FN(ProfiledType::affine_grid_generator_backward));
  m.impl("alias", TORCH_FN(ProfiledType::alias));
  m.impl("align_as", TORCH_FN(ProfiledType::align_as));
  m.impl("align_tensors", TORCH_FN(ProfiledType::align_tensors));
  m.impl_UNBOXED("align_to", &ProfiledType::align_to);
  m.impl_UNBOXED("align_to.ellipsis_idx", &ProfiledType::align_to_ellipsis_idx);
  m.impl("all.dim", TORCH_FN(ProfiledType::all_dim));
  m.impl_UNBOXED("all.dimname", &ProfiledType::all_dimname);
  m.impl("all", TORCH_FN(ProfiledType::all));
  m.impl_UNBOXED("all.out", &ProfiledType::all_out_out);
  m.impl_UNBOXED("all.dimname_out", &ProfiledType::all_out_dimname_out);
  m.impl("allclose", TORCH_FN(ProfiledType::allclose));
  m.impl("alpha_dropout", TORCH_FN(ProfiledType::alpha_dropout));
  m.impl_UNBOXED("alpha_dropout_", &ProfiledType::alpha_dropout_);
  m.impl("angle", TORCH_FN(ProfiledType::angle));
  m.impl_UNBOXED("angle.out", &ProfiledType::angle_out_out);
  m.impl("any.dim", TORCH_FN(ProfiledType::any_dim));
  m.impl_UNBOXED("any.dimname", &ProfiledType::any_dimname);
  m.impl("any", TORCH_FN(ProfiledType::any));
  m.impl_UNBOXED("any.out", &ProfiledType::any_out_out);
  m.impl_UNBOXED("any.dimname_out", &ProfiledType::any_out_dimname_out);
  m.impl_UNBOXED("arange", &ProfiledType::arange);
  m.impl_UNBOXED("arange.start", &ProfiledType::arange_start);
  m.impl_UNBOXED("arange.start_step", &ProfiledType::arange_start_step);
  m.impl_UNBOXED("arange.out", &ProfiledType::arange_out_out);
  m.impl_UNBOXED("arange.start_out", &ProfiledType::arange_out_start_out);
  m.impl("argmax", TORCH_FN(ProfiledType::argmax));
  m.impl("argmin", TORCH_FN(ProfiledType::argmin));
  m.impl("argsort", TORCH_FN(ProfiledType::argsort));
  m.impl_UNBOXED("argsort.dimname", &ProfiledType::argsort_dimname);
  m.impl("as_strided", TORCH_FN(ProfiledType::as_strided));
  m.impl_UNBOXED("as_strided_", &ProfiledType::as_strided_);
  m.impl("asin", TORCH_FN(ProfiledType::asin));
  m.impl_UNBOXED("asin_", &ProfiledType::asin_);
  m.impl_UNBOXED("asin.out", &ProfiledType::asin_out_out);
  m.impl("asinh", TORCH_FN(ProfiledType::asinh));
  m.impl_UNBOXED("asinh_", &ProfiledType::asinh_);
  m.impl_UNBOXED("asinh.out", &ProfiledType::asinh_out_out);
  m.impl("atan", TORCH_FN(ProfiledType::atan));
  m.impl("atan2", TORCH_FN(ProfiledType::atan2));
  m.impl_UNBOXED("atan2_", &ProfiledType::atan2_);
  m.impl_UNBOXED("atan2.out", &ProfiledType::atan2_out_out);
  m.impl_UNBOXED("atan_", &ProfiledType::atan_);
  m.impl_UNBOXED("atan.out", &ProfiledType::atan_out_out);
  m.impl("atanh", TORCH_FN(ProfiledType::atanh));
  m.impl_UNBOXED("atanh_", &ProfiledType::atanh_);
  m.impl_UNBOXED("atanh.out", &ProfiledType::atanh_out_out);
  m.impl("avg_pool1d", TORCH_FN(ProfiledType::avg_pool1d));
  m.impl("avg_pool2d", TORCH_FN(ProfiledType::avg_pool2d));
  m.impl("avg_pool2d_backward", TORCH_FN(ProfiledType::avg_pool2d_backward));
  m.impl_UNBOXED("avg_pool2d_backward.grad_input", &ProfiledType::avg_pool2d_backward_out_grad_input);
  m.impl_UNBOXED("avg_pool2d.out", &ProfiledType::avg_pool2d_out_out);
  m.impl("avg_pool3d", TORCH_FN(ProfiledType::avg_pool3d));
  m.impl("avg_pool3d_backward", TORCH_FN(ProfiledType::avg_pool3d_backward));
  m.impl_UNBOXED("avg_pool3d_backward.grad_input", &ProfiledType::avg_pool3d_backward_out_grad_input);
  m.impl_UNBOXED("avg_pool3d.out", &ProfiledType::avg_pool3d_out_out);
  m.impl_UNBOXED("backward", &ProfiledType::backward);
  m.impl("baddbmm", TORCH_FN(ProfiledType::baddbmm));
  m.impl_UNBOXED("baddbmm_", &ProfiledType::baddbmm_);
  m.impl_UNBOXED("baddbmm.out", &ProfiledType::baddbmm_out_out);
  m.impl_UNBOXED("bartlett_window", &ProfiledType::bartlett_window);
  m.impl_UNBOXED("bartlett_window.periodic", &ProfiledType::bartlett_window_periodic);
  m.impl_UNBOXED("batch_norm", &ProfiledType::batch_norm);
  m.impl_UNBOXED("batch_norm_backward_elemt", &ProfiledType::batch_norm_backward_elemt);
  m.impl_UNBOXED("batch_norm_backward_reduce", &ProfiledType::batch_norm_backward_reduce);
  m.impl_UNBOXED("batch_norm_elemt", &ProfiledType::batch_norm_elemt);
  m.impl_UNBOXED("batch_norm_elemt.out", &ProfiledType::batch_norm_elemt_out_out);
  m.impl_UNBOXED("batch_norm_gather_stats", &ProfiledType::batch_norm_gather_stats);
  m.impl_UNBOXED("batch_norm_gather_stats_with_counts", &ProfiledType::batch_norm_gather_stats_with_counts);
  m.impl("batch_norm_stats", TORCH_FN(ProfiledType::batch_norm_stats));
  m.impl_UNBOXED("batch_norm_update_stats", &ProfiledType::batch_norm_update_stats);
  m.impl_UNBOXED("bernoulli", &ProfiledType::bernoulli);
  m.impl_UNBOXED("bernoulli.p", &ProfiledType::bernoulli_p);
  m.impl_UNBOXED("bernoulli_.Tensor", &ProfiledType::bernoulli__Tensor);
  m.impl_UNBOXED("bernoulli_.float", &ProfiledType::bernoulli__float);
  m.impl_UNBOXED("bernoulli.out", &ProfiledType::bernoulli_out_out);
  m.impl_UNBOXED("bilinear", &ProfiledType::bilinear);
  m.impl_UNBOXED("binary_cross_entropy", &ProfiledType::binary_cross_entropy);
  m.impl_UNBOXED("binary_cross_entropy_backward", &ProfiledType::binary_cross_entropy_backward);
  m.impl_UNBOXED("binary_cross_entropy_backward.grad_input", &ProfiledType::binary_cross_entropy_backward_out_grad_input);
  m.impl_UNBOXED("binary_cross_entropy.out", &ProfiledType::binary_cross_entropy_out_out);
  m.impl_UNBOXED("binary_cross_entropy_with_logits", &ProfiledType::binary_cross_entropy_with_logits);
  m.impl_UNBOXED("binary_cross_entropy_with_logits_backward", &ProfiledType::binary_cross_entropy_with_logits_backward);
  m.impl_UNBOXED("bincount", &ProfiledType::bincount);
  m.impl_UNBOXED("binomial", &ProfiledType::binomial);
  m.impl("bitwise_and.Scalar", TORCH_FN(ProfiledType::bitwise_and_Scalar));
  m.impl("bitwise_and.Tensor", TORCH_FN(ProfiledType::bitwise_and_Tensor));
  m.impl_UNBOXED("bitwise_and_.Scalar", &ProfiledType::bitwise_and__Scalar);
  m.impl_UNBOXED("bitwise_and_.Tensor", &ProfiledType::bitwise_and__Tensor);
  m.impl_UNBOXED("bitwise_and.Tensor_out", &ProfiledType::bitwise_and_out_Tensor_out);
  m.impl_UNBOXED("bitwise_and.Scalar_out", &ProfiledType::bitwise_and_out_Scalar_out);
  m.impl("bitwise_not", TORCH_FN(ProfiledType::bitwise_not));
  m.impl_UNBOXED("bitwise_not_", &ProfiledType::bitwise_not_);
  m.impl_UNBOXED("bitwise_not.out", &ProfiledType::bitwise_not_out_out);
  m.impl("bitwise_or.Scalar", TORCH_FN(ProfiledType::bitwise_or_Scalar));
  m.impl("bitwise_or.Tensor", TORCH_FN(ProfiledType::bitwise_or_Tensor));
  m.impl_UNBOXED("bitwise_or_.Scalar", &ProfiledType::bitwise_or__Scalar);
  m.impl_UNBOXED("bitwise_or_.Tensor", &ProfiledType::bitwise_or__Tensor);
  m.impl_UNBOXED("bitwise_or.Tensor_out", &ProfiledType::bitwise_or_out_Tensor_out);
  m.impl_UNBOXED("bitwise_or.Scalar_out", &ProfiledType::bitwise_or_out_Scalar_out);
  m.impl("bitwise_xor.Scalar", TORCH_FN(ProfiledType::bitwise_xor_Scalar));
  m.impl("bitwise_xor.Tensor", TORCH_FN(ProfiledType::bitwise_xor_Tensor));
  m.impl_UNBOXED("bitwise_xor_.Scalar", &ProfiledType::bitwise_xor__Scalar);
  m.impl_UNBOXED("bitwise_xor_.Tensor", &ProfiledType::bitwise_xor__Tensor);
  m.impl_UNBOXED("bitwise_xor.Tensor_out", &ProfiledType::bitwise_xor_out_Tensor_out);
  m.impl_UNBOXED("bitwise_xor.Scalar_out", &ProfiledType::bitwise_xor_out_Scalar_out);
  m.impl_UNBOXED("blackman_window", &ProfiledType::blackman_window);
  m.impl_UNBOXED("blackman_window.periodic", &ProfiledType::blackman_window_periodic);
  m.impl("block_diag", TORCH_FN(ProfiledType::block_diag));
  m.impl("bmm", TORCH_FN(ProfiledType::bmm));
  m.impl_UNBOXED("bmm.out", &ProfiledType::bmm_out_out);
  m.impl("broadcast_tensors", TORCH_FN(ProfiledType::broadcast_tensors));
  m.impl("bucketize.Tensor", TORCH_FN(ProfiledType::bucketize_Tensor));
  m.impl("bucketize.Scalar", TORCH_FN(ProfiledType::bucketize_Scalar));
  m.impl_UNBOXED("bucketize.Tensor_out", &ProfiledType::bucketize_out_Tensor_out);
  m.impl_UNBOXED("can_cast", &ProfiledType::can_cast);
  m.impl("cartesian_prod", TORCH_FN(ProfiledType::cartesian_prod));
  m.impl("cat", TORCH_FN(ProfiledType::cat));
  m.impl_UNBOXED("cat.names", &ProfiledType::cat_names);
  m.impl_UNBOXED("cat.out", &ProfiledType::cat_out_out);
  m.impl_UNBOXED("cat.names_out", &ProfiledType::cat_out_names_out);
  m.impl_UNBOXED("cauchy_", &ProfiledType::cauchy_);
  m.impl("cdist", TORCH_FN(ProfiledType::cdist));
  m.impl("ceil", TORCH_FN(ProfiledType::ceil));
  m.impl_UNBOXED("ceil_", &ProfiledType::ceil_);
  m.impl_UNBOXED("ceil.out", &ProfiledType::ceil_out_out);
  m.impl("celu", TORCH_FN(ProfiledType::celu));
  m.impl_UNBOXED("celu_", &ProfiledType::celu_);
  m.impl("chain_matmul", TORCH_FN(ProfiledType::chain_matmul));
  m.impl("channel_shuffle", TORCH_FN(ProfiledType::channel_shuffle));
  m.impl("cholesky", TORCH_FN(ProfiledType::cholesky));
  m.impl("cholesky_inverse", TORCH_FN(ProfiledType::cholesky_inverse));
  m.impl_UNBOXED("cholesky_inverse.out", &ProfiledType::cholesky_inverse_out_out);
  m.impl_UNBOXED("cholesky.out", &ProfiledType::cholesky_out_out);
  m.impl("cholesky_solve", TORCH_FN(ProfiledType::cholesky_solve));
  m.impl_UNBOXED("cholesky_solve.out", &ProfiledType::cholesky_solve_out_out);
  m.impl("chunk", TORCH_FN(ProfiledType::chunk));
  m.impl("clamp", TORCH_FN(ProfiledType::clamp));
  m.impl_UNBOXED("clamp_", &ProfiledType::clamp_);
  m.impl("clamp_max", TORCH_FN(ProfiledType::clamp_max));
  m.impl_UNBOXED("clamp_max_", &ProfiledType::clamp_max_);
  m.impl_UNBOXED("clamp_max.out", &ProfiledType::clamp_max_out_out);
  m.impl("clamp_min", TORCH_FN(ProfiledType::clamp_min));
  m.impl_UNBOXED("clamp_min_", &ProfiledType::clamp_min_);
  m.impl_UNBOXED("clamp_min.out", &ProfiledType::clamp_min_out_out);
  m.impl_UNBOXED("clamp.out", &ProfiledType::clamp_out_out);
  m.impl_UNBOXED("clone", &ProfiledType::clone);
  m.impl("coalesce", TORCH_FN(ProfiledType::coalesce));
  m.impl("col2im", TORCH_FN(ProfiledType::col2im));
  m.impl("col2im_backward", TORCH_FN(ProfiledType::col2im_backward));
  m.impl_UNBOXED("col2im_backward.grad_input", &ProfiledType::col2im_backward_out_grad_input);
  m.impl_UNBOXED("col2im.out", &ProfiledType::col2im_out_out);
  m.impl("combinations", TORCH_FN(ProfiledType::combinations));
  m.impl("conj", TORCH_FN(ProfiledType::conj));
  m.impl_UNBOXED("conj.out", &ProfiledType::conj_out_out);
  m.impl("constant_pad_nd", TORCH_FN(ProfiledType::constant_pad_nd));
  m.impl_UNBOXED("contiguous", &ProfiledType::contiguous);
  m.impl_UNBOXED("conv1d", &ProfiledType::conv1d);
  m.impl_UNBOXED("conv2d", &ProfiledType::conv2d);
  m.impl_UNBOXED("conv3d", &ProfiledType::conv3d);
  m.impl("conv_tbc", TORCH_FN(ProfiledType::conv_tbc));
  m.impl("conv_tbc_backward", TORCH_FN(ProfiledType::conv_tbc_backward));
  m.impl_UNBOXED("conv_transpose1d", &ProfiledType::conv_transpose1d);
  m.impl_UNBOXED("conv_transpose2d.input", &ProfiledType::conv_transpose2d_input);
  m.impl_UNBOXED("conv_transpose3d.input", &ProfiledType::conv_transpose3d_input);
  m.impl_UNBOXED("convolution", &ProfiledType::convolution);
  m.impl("convolution_backward_overrideable", TORCH_FN(ProfiledType::convolution_backward_overrideable));
  m.impl_UNBOXED("convolution_overrideable", &ProfiledType::convolution_overrideable);
  m.impl_UNBOXED("copy_", &ProfiledType::copy_);
  m.impl_UNBOXED("copy_sparse_to_sparse_", &ProfiledType::copy_sparse_to_sparse_);
  m.impl("cos", TORCH_FN(ProfiledType::cos));
  m.impl_UNBOXED("cos_", &ProfiledType::cos_);
  m.impl_UNBOXED("cos.out", &ProfiledType::cos_out_out);
  m.impl("cosh", TORCH_FN(ProfiledType::cosh));
  m.impl_UNBOXED("cosh_", &ProfiledType::cosh_);
  m.impl_UNBOXED("cosh.out", &ProfiledType::cosh_out_out);
  m.impl("cosine_embedding_loss", TORCH_FN(ProfiledType::cosine_embedding_loss));
  m.impl("cosine_similarity", TORCH_FN(ProfiledType::cosine_similarity));
  m.impl("cross", TORCH_FN(ProfiledType::cross));
  m.impl_UNBOXED("cross.out", &ProfiledType::cross_out_out);
  m.impl("ctc_loss.IntList", TORCH_FN(ProfiledType::ctc_loss_IntList));
  m.impl("ctc_loss.Tensor", TORCH_FN(ProfiledType::ctc_loss_Tensor));
  m.impl("cudnn_affine_grid_generator", TORCH_FN(ProfiledType::cudnn_affine_grid_generator));
  m.impl("cudnn_affine_grid_generator_backward", TORCH_FN(ProfiledType::cudnn_affine_grid_generator_backward));
  m.impl_UNBOXED("cudnn_batch_norm", &ProfiledType::cudnn_batch_norm);
  m.impl_UNBOXED("cudnn_batch_norm_backward", &ProfiledType::cudnn_batch_norm_backward);
  m.impl_UNBOXED("cudnn_convolution.deprecated", &ProfiledType::cudnn_convolution_deprecated);
  m.impl("cudnn_convolution", TORCH_FN(ProfiledType::cudnn_convolution));
  m.impl("cudnn_convolution_backward", TORCH_FN(ProfiledType::cudnn_convolution_backward));
  m.impl("cudnn_convolution_backward_input", TORCH_FN(ProfiledType::cudnn_convolution_backward_input));
  m.impl("cudnn_convolution_backward_weight", TORCH_FN(ProfiledType::cudnn_convolution_backward_weight));
  m.impl_UNBOXED("cudnn_convolution_transpose.deprecated", &ProfiledType::cudnn_convolution_transpose_deprecated);
  m.impl("cudnn_convolution_transpose", TORCH_FN(ProfiledType::cudnn_convolution_transpose));
  m.impl("cudnn_convolution_transpose_backward", TORCH_FN(ProfiledType::cudnn_convolution_transpose_backward));
  m.impl("cudnn_convolution_transpose_backward_input", TORCH_FN(ProfiledType::cudnn_convolution_transpose_backward_input));
  m.impl("cudnn_convolution_transpose_backward_weight", TORCH_FN(ProfiledType::cudnn_convolution_transpose_backward_weight));
  m.impl("cudnn_grid_sampler", TORCH_FN(ProfiledType::cudnn_grid_sampler));
  m.impl("cudnn_grid_sampler_backward", TORCH_FN(ProfiledType::cudnn_grid_sampler_backward));
  m.impl("cudnn_is_acceptable", TORCH_FN(ProfiledType::cudnn_is_acceptable));
  m.impl("cummax", TORCH_FN(ProfiledType::cummax));
  m.impl_UNBOXED("cummax.dimname", &ProfiledType::cummax_dimname);
  m.impl_UNBOXED("cummax.out", &ProfiledType::cummax_out_out);
  m.impl_UNBOXED("cummax.dimname_out", &ProfiledType::cummax_out_dimname_out);
  m.impl("cummin", TORCH_FN(ProfiledType::cummin));
  m.impl_UNBOXED("cummin.dimname", &ProfiledType::cummin_dimname);
  m.impl_UNBOXED("cummin.out", &ProfiledType::cummin_out_out);
  m.impl_UNBOXED("cummin.dimname_out", &ProfiledType::cummin_out_dimname_out);
  m.impl_UNBOXED("cumprod", &ProfiledType::cumprod);
  m.impl_UNBOXED("cumprod.dimname", &ProfiledType::cumprod_dimname);
  m.impl_UNBOXED("cumprod.out", &ProfiledType::cumprod_out_out);
  m.impl_UNBOXED("cumprod.dimname_out", &ProfiledType::cumprod_out_dimname_out);
  m.impl_UNBOXED("cumsum", &ProfiledType::cumsum);
  m.impl_UNBOXED("cumsum.dimname", &ProfiledType::cumsum_dimname);
  m.impl_UNBOXED("cumsum.out", &ProfiledType::cumsum_out_out);
  m.impl_UNBOXED("cumsum.dimname_out", &ProfiledType::cumsum_out_dimname_out);
  m.impl("data", TORCH_FN(ProfiledType::data));
  m.impl("deg2rad", TORCH_FN(ProfiledType::deg2rad));
  m.impl_UNBOXED("deg2rad_", &ProfiledType::deg2rad_);
  m.impl_UNBOXED("deg2rad.out", &ProfiledType::deg2rad_out_out);
  m.impl("dense_dim", TORCH_FN(ProfiledType::dense_dim));
  m.impl("dequantize.self", TORCH_FN(ProfiledType::dequantize_self));
  m.impl("dequantize.tensors", TORCH_FN(ProfiledType::dequantize_tensors));
  m.impl("det", TORCH_FN(ProfiledType::det));
  m.impl("detach", TORCH_FN(ProfiledType::detach));
  m.impl_UNBOXED("detach_", &ProfiledType::detach_);
  m.impl("diag", TORCH_FN(ProfiledType::diag));
  m.impl("diag_embed", TORCH_FN(ProfiledType::diag_embed));
  m.impl_UNBOXED("diag.out", &ProfiledType::diag_out_out);
  m.impl("diagflat", TORCH_FN(ProfiledType::diagflat));
  m.impl("diagonal", TORCH_FN(ProfiledType::diagonal));
  m.impl_UNBOXED("diagonal.Dimname", &ProfiledType::diagonal_Dimname);
  m.impl("digamma", TORCH_FN(ProfiledType::digamma));
  m.impl_UNBOXED("digamma_", &ProfiledType::digamma_);
  m.impl_UNBOXED("digamma.out", &ProfiledType::digamma_out_out);
  m.impl("dist", TORCH_FN(ProfiledType::dist));
  m.impl("div.Tensor", TORCH_FN(ProfiledType::div_Tensor));
  m.impl("div.Scalar", TORCH_FN(ProfiledType::div_Scalar));
  m.impl_UNBOXED("div_.Tensor", &ProfiledType::div__Tensor);
  m.impl_UNBOXED("div_.Scalar", &ProfiledType::div__Scalar);
  m.impl_UNBOXED("div.out", &ProfiledType::div_out_out);
  m.impl("dot", TORCH_FN(ProfiledType::dot));
  m.impl_UNBOXED("dot.out", &ProfiledType::dot_out_out);
  m.impl("dropout", TORCH_FN(ProfiledType::dropout));
  m.impl_UNBOXED("dropout_", &ProfiledType::dropout_);
  m.impl("eig", TORCH_FN(ProfiledType::eig));
  m.impl_UNBOXED("eig.e", &ProfiledType::eig_out_e);
  m.impl("einsum", TORCH_FN(ProfiledType::einsum));
  m.impl("elu", TORCH_FN(ProfiledType::elu));
  m.impl_UNBOXED("elu_", &ProfiledType::elu_);
  m.impl("elu_backward", TORCH_FN(ProfiledType::elu_backward));
  m.impl_UNBOXED("elu_backward.grad_input", &ProfiledType::elu_backward_out_grad_input);
  m.impl_UNBOXED("elu.out", &ProfiledType::elu_out_out);
  m.impl("embedding", TORCH_FN(ProfiledType::embedding));
  m.impl("embedding_backward", TORCH_FN(ProfiledType::embedding_backward));
  m.impl_UNBOXED("embedding_bag", &ProfiledType::embedding_bag);
  m.impl("embedding_dense_backward", TORCH_FN(ProfiledType::embedding_dense_backward));
  m.impl_UNBOXED("embedding_renorm_", &ProfiledType::embedding_renorm_);
  m.impl("embedding_sparse_backward", TORCH_FN(ProfiledType::embedding_sparse_backward));
  m.impl_UNBOXED("empty.names", &ProfiledType::empty_names);
  m.impl_UNBOXED("empty.memory_format", &ProfiledType::empty_memory_format);
  m.impl_UNBOXED("empty_like", &ProfiledType::empty_like);
  m.impl_UNBOXED("empty_meta", &ProfiledType::empty_meta);
  m.impl_UNBOXED("empty.out", &ProfiledType::empty_out_out);
  m.impl_UNBOXED("empty_quantized", &ProfiledType::empty_quantized);
  m.impl_UNBOXED("empty_strided", &ProfiledType::empty_strided);
  m.impl("eq.Scalar", TORCH_FN(ProfiledType::eq_Scalar));
  m.impl("eq.Tensor", TORCH_FN(ProfiledType::eq_Tensor));
  m.impl_UNBOXED("eq_.Scalar", &ProfiledType::eq__Scalar);
  m.impl_UNBOXED("eq_.Tensor", &ProfiledType::eq__Tensor);
  m.impl_UNBOXED("eq.Scalar_out", &ProfiledType::eq_out_Scalar_out);
  m.impl_UNBOXED("eq.Tensor_out", &ProfiledType::eq_out_Tensor_out);
  m.impl("equal", TORCH_FN(ProfiledType::equal));
  m.impl("erf", TORCH_FN(ProfiledType::erf));
  m.impl_UNBOXED("erf_", &ProfiledType::erf_);
  m.impl_UNBOXED("erf.out", &ProfiledType::erf_out_out);
  m.impl("erfc", TORCH_FN(ProfiledType::erfc));
  m.impl_UNBOXED("erfc_", &ProfiledType::erfc_);
  m.impl_UNBOXED("erfc.out", &ProfiledType::erfc_out_out);
  m.impl("erfinv", TORCH_FN(ProfiledType::erfinv));
  m.impl_UNBOXED("erfinv_", &ProfiledType::erfinv_);
  m.impl_UNBOXED("erfinv.out", &ProfiledType::erfinv_out_out);
  m.impl("exp", TORCH_FN(ProfiledType::exp));
  m.impl_UNBOXED("exp_", &ProfiledType::exp_);
  m.impl_UNBOXED("exp.out", &ProfiledType::exp_out_out);
  m.impl("expand", TORCH_FN(ProfiledType::expand));
  m.impl("expand_as", TORCH_FN(ProfiledType::expand_as));
  m.impl("expm1", TORCH_FN(ProfiledType::expm1));
  m.impl_UNBOXED("expm1_", &ProfiledType::expm1_);
  m.impl_UNBOXED("expm1.out", &ProfiledType::expm1_out_out);
  m.impl_UNBOXED("exponential_", &ProfiledType::exponential_);
  m.impl_UNBOXED("eye", &ProfiledType::eye);
  m.impl_UNBOXED("eye.m", &ProfiledType::eye_m);
  m.impl_UNBOXED("eye.out", &ProfiledType::eye_out_out);
  m.impl_UNBOXED("eye.m_out", &ProfiledType::eye_out_m_out);
  m.impl("fake_quantize_per_channel_affine", TORCH_FN(ProfiledType::fake_quantize_per_channel_affine));
  m.impl("fake_quantize_per_channel_affine_backward", TORCH_FN(ProfiledType::fake_quantize_per_channel_affine_backward));
  m.impl("fake_quantize_per_tensor_affine", TORCH_FN(ProfiledType::fake_quantize_per_tensor_affine));
  m.impl("fake_quantize_per_tensor_affine_backward", TORCH_FN(ProfiledType::fake_quantize_per_tensor_affine_backward));
  m.impl("fbgemm_linear_fp16_weight", TORCH_FN(ProfiledType::fbgemm_linear_fp16_weight));
  m.impl("fbgemm_linear_fp16_weight_fp32_activation", TORCH_FN(ProfiledType::fbgemm_linear_fp16_weight_fp32_activation));
  m.impl("fbgemm_linear_int8_weight", TORCH_FN(ProfiledType::fbgemm_linear_int8_weight));
  m.impl("fbgemm_linear_int8_weight_fp32_activation", TORCH_FN(ProfiledType::fbgemm_linear_int8_weight_fp32_activation));
  m.impl("fbgemm_linear_quantize_weight", TORCH_FN(ProfiledType::fbgemm_linear_quantize_weight));
  m.impl("fbgemm_pack_gemm_matrix_fp16", TORCH_FN(ProfiledType::fbgemm_pack_gemm_matrix_fp16));
  m.impl("fbgemm_pack_quantized_matrix", TORCH_FN(ProfiledType::fbgemm_pack_quantized_matrix));
  m.impl("fbgemm_pack_quantized_matrix.KN", TORCH_FN(ProfiledType::fbgemm_pack_quantized_matrix_KN));
  m.impl("feature_alpha_dropout", TORCH_FN(ProfiledType::feature_alpha_dropout));
  m.impl_UNBOXED("feature_alpha_dropout_", &ProfiledType::feature_alpha_dropout_);
  m.impl("feature_dropout", TORCH_FN(ProfiledType::feature_dropout));
  m.impl_UNBOXED("feature_dropout_", &ProfiledType::feature_dropout_);
  m.impl("fft", TORCH_FN(ProfiledType::fft));
  m.impl_UNBOXED("fill_.Scalar", &ProfiledType::fill__Scalar);
  m.impl_UNBOXED("fill_.Tensor", &ProfiledType::fill__Tensor);
  m.impl_UNBOXED("fill_diagonal_", &ProfiledType::fill_diagonal_);
  m.impl("flatten.using_ints", TORCH_FN(ProfiledType::flatten_using_ints));
  m.impl_UNBOXED("flatten.named_out_dim", &ProfiledType::flatten_named_out_dim);
  m.impl_UNBOXED("flatten.using_names", &ProfiledType::flatten_using_names);
  m.impl_UNBOXED("flatten.DimnameList", &ProfiledType::flatten_DimnameList);
  m.impl("flip", TORCH_FN(ProfiledType::flip));
  m.impl("fliplr", TORCH_FN(ProfiledType::fliplr));
  m.impl("flipud", TORCH_FN(ProfiledType::flipud));
  m.impl("floor", TORCH_FN(ProfiledType::floor));
  m.impl_UNBOXED("floor_", &ProfiledType::floor_);
  m.impl("floor_divide", TORCH_FN(ProfiledType::floor_divide));
  m.impl("floor_divide.Scalar", TORCH_FN(ProfiledType::floor_divide_Scalar));
  m.impl_UNBOXED("floor_divide_.Tensor", &ProfiledType::floor_divide__Tensor);
  m.impl_UNBOXED("floor_divide_.Scalar", &ProfiledType::floor_divide__Scalar);
  m.impl_UNBOXED("floor_divide.out", &ProfiledType::floor_divide_out_out);
  m.impl_UNBOXED("floor.out", &ProfiledType::floor_out_out);
  m.impl("fmod.Scalar", TORCH_FN(ProfiledType::fmod_Scalar));
  m.impl("fmod.Tensor", TORCH_FN(ProfiledType::fmod_Tensor));
  m.impl_UNBOXED("fmod_.Scalar", &ProfiledType::fmod__Scalar);
  m.impl_UNBOXED("fmod_.Tensor", &ProfiledType::fmod__Tensor);
  m.impl_UNBOXED("fmod.Scalar_out", &ProfiledType::fmod_out_Scalar_out);
  m.impl_UNBOXED("fmod.Tensor_out", &ProfiledType::fmod_out_Tensor_out);
  m.impl("frac", TORCH_FN(ProfiledType::frac));
  m.impl_UNBOXED("frac_", &ProfiledType::frac_);
  m.impl_UNBOXED("frac.out", &ProfiledType::frac_out_out);
  m.impl("fractional_max_pool2d", TORCH_FN(ProfiledType::fractional_max_pool2d));
  m.impl("fractional_max_pool2d_backward", TORCH_FN(ProfiledType::fractional_max_pool2d_backward));
  m.impl_UNBOXED("fractional_max_pool2d_backward.grad_input", &ProfiledType::fractional_max_pool2d_backward_out_grad_input);
  m.impl_UNBOXED("fractional_max_pool2d.output", &ProfiledType::fractional_max_pool2d_out_output);
  m.impl("fractional_max_pool3d", TORCH_FN(ProfiledType::fractional_max_pool3d));
  m.impl("fractional_max_pool3d_backward", TORCH_FN(ProfiledType::fractional_max_pool3d_backward));
  m.impl_UNBOXED("fractional_max_pool3d_backward.grad_input", &ProfiledType::fractional_max_pool3d_backward_out_grad_input);
  m.impl_UNBOXED("fractional_max_pool3d.output", &ProfiledType::fractional_max_pool3d_out_output);
  m.impl("frobenius_norm", TORCH_FN(ProfiledType::frobenius_norm));
  m.impl("frobenius_norm.dim", TORCH_FN(ProfiledType::frobenius_norm_dim));
  m.impl_UNBOXED("frobenius_norm.out", &ProfiledType::frobenius_norm_out_out);
  m.impl_UNBOXED("from_file", &ProfiledType::from_file);
  m.impl_UNBOXED("full.names", &ProfiledType::full_names);
  m.impl_UNBOXED("full", &ProfiledType::full);
  m.impl_UNBOXED("full_like", &ProfiledType::full_like);
  m.impl_UNBOXED("full.out", &ProfiledType::full_out_out);
  m.impl("gather", TORCH_FN(ProfiledType::gather));
  m.impl_UNBOXED("gather.dimname", &ProfiledType::gather_dimname);
  m.impl_UNBOXED("gather.out", &ProfiledType::gather_out_out);
  m.impl_UNBOXED("gather.dimname_out", &ProfiledType::gather_out_dimname_out);
  m.impl("ge.Scalar", TORCH_FN(ProfiledType::ge_Scalar));
  m.impl("ge.Tensor", TORCH_FN(ProfiledType::ge_Tensor));
  m.impl_UNBOXED("ge_.Scalar", &ProfiledType::ge__Scalar);
  m.impl_UNBOXED("ge_.Tensor", &ProfiledType::ge__Tensor);
  m.impl_UNBOXED("ge.Scalar_out", &ProfiledType::ge_out_Scalar_out);
  m.impl_UNBOXED("ge.Tensor_out", &ProfiledType::ge_out_Tensor_out);
  m.impl("gelu", TORCH_FN(ProfiledType::gelu));
  m.impl("gelu_backward", TORCH_FN(ProfiledType::gelu_backward));
  m.impl_UNBOXED("geometric_", &ProfiledType::geometric_);
  m.impl("geqrf", TORCH_FN(ProfiledType::geqrf));
  m.impl_UNBOXED("geqrf.a", &ProfiledType::geqrf_out_a);
  m.impl("ger", TORCH_FN(ProfiledType::ger));
  m.impl_UNBOXED("ger.out", &ProfiledType::ger_out_out);
  m.impl("glu", TORCH_FN(ProfiledType::glu));
  m.impl("glu_backward", TORCH_FN(ProfiledType::glu_backward));
  m.impl_UNBOXED("glu_backward.grad_input", &ProfiledType::glu_backward_out_grad_input);
  m.impl_UNBOXED("glu.out", &ProfiledType::glu_out_out);
  m.impl("grid_sampler", TORCH_FN(ProfiledType::grid_sampler));
  m.impl("grid_sampler_2d", TORCH_FN(ProfiledType::grid_sampler_2d));
  m.impl("grid_sampler_2d_backward", TORCH_FN(ProfiledType::grid_sampler_2d_backward));
  m.impl("grid_sampler_3d", TORCH_FN(ProfiledType::grid_sampler_3d));
  m.impl("grid_sampler_3d_backward", TORCH_FN(ProfiledType::grid_sampler_3d_backward));
  m.impl_UNBOXED("group_norm", &ProfiledType::group_norm);
  m.impl("gru.input", TORCH_FN(ProfiledType::gru_input));
  m.impl("gru.data", TORCH_FN(ProfiledType::gru_data));
  m.impl_UNBOXED("gru_cell", &ProfiledType::gru_cell);
  m.impl("gt.Scalar", TORCH_FN(ProfiledType::gt_Scalar));
  m.impl("gt.Tensor", TORCH_FN(ProfiledType::gt_Tensor));
  m.impl_UNBOXED("gt_.Scalar", &ProfiledType::gt__Scalar);
  m.impl_UNBOXED("gt_.Tensor", &ProfiledType::gt__Tensor);
  m.impl_UNBOXED("gt.Scalar_out", &ProfiledType::gt_out_Scalar_out);
  m.impl_UNBOXED("gt.Tensor_out", &ProfiledType::gt_out_Tensor_out);
  m.impl_UNBOXED("hamming_window", &ProfiledType::hamming_window);
  m.impl_UNBOXED("hamming_window.periodic", &ProfiledType::hamming_window_periodic);
  m.impl_UNBOXED("hamming_window.periodic_alpha", &ProfiledType::hamming_window_periodic_alpha);
  m.impl_UNBOXED("hamming_window.periodic_alpha_beta", &ProfiledType::hamming_window_periodic_alpha_beta);
  m.impl_UNBOXED("hann_window", &ProfiledType::hann_window);
  m.impl_UNBOXED("hann_window.periodic", &ProfiledType::hann_window_periodic);
  m.impl("hardshrink", TORCH_FN(ProfiledType::hardshrink));
  m.impl("hardshrink_backward", TORCH_FN(ProfiledType::hardshrink_backward));
  m.impl("hardsigmoid", TORCH_FN(ProfiledType::hardsigmoid));
  m.impl_UNBOXED("hardsigmoid_", &ProfiledType::hardsigmoid_);
  m.impl("hardsigmoid_backward", TORCH_FN(ProfiledType::hardsigmoid_backward));
  m.impl_UNBOXED("hardsigmoid.out", &ProfiledType::hardsigmoid_out_out);
  m.impl("hardswish", TORCH_FN(ProfiledType::hardswish));
  m.impl_UNBOXED("hardswish_", &ProfiledType::hardswish_);
  m.impl("hardswish_backward", TORCH_FN(ProfiledType::hardswish_backward));
  m.impl_UNBOXED("hardswish.out", &ProfiledType::hardswish_out_out);
  m.impl("hardtanh", TORCH_FN(ProfiledType::hardtanh));
  m.impl_UNBOXED("hardtanh_", &ProfiledType::hardtanh_);
  m.impl("hardtanh_backward", TORCH_FN(ProfiledType::hardtanh_backward));
  m.impl_UNBOXED("hardtanh_backward.grad_input", &ProfiledType::hardtanh_backward_out_grad_input);
  m.impl_UNBOXED("hardtanh.out", &ProfiledType::hardtanh_out_out);
  m.impl("hinge_embedding_loss", TORCH_FN(ProfiledType::hinge_embedding_loss));
  m.impl("histc", TORCH_FN(ProfiledType::histc));
  m.impl_UNBOXED("histc.out", &ProfiledType::histc_out_out);
  m.impl("hspmm", TORCH_FN(ProfiledType::hspmm));
  m.impl_UNBOXED("hspmm.out", &ProfiledType::hspmm_out_out);
  m.impl("ifft", TORCH_FN(ProfiledType::ifft));
  m.impl("im2col", TORCH_FN(ProfiledType::im2col));
  m.impl("im2col_backward", TORCH_FN(ProfiledType::im2col_backward));
  m.impl_UNBOXED("im2col_backward.grad_input", &ProfiledType::im2col_backward_out_grad_input);
  m.impl_UNBOXED("im2col.out", &ProfiledType::im2col_out_out);
  m.impl("imag", TORCH_FN(ProfiledType::imag));
  m.impl_UNBOXED("index.Tensor", &ProfiledType::index_Tensor);
  m.impl("index_add", TORCH_FN(ProfiledType::index_add));
  m.impl_UNBOXED("index_add.dimname", &ProfiledType::index_add_dimname);
  m.impl_UNBOXED("index_add_", &ProfiledType::index_add_);
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
  m.impl_UNBOXED("index_put", &ProfiledType::index_put);
  m.impl_UNBOXED("index_put_", &ProfiledType::index_put_);
  m.impl("index_select", TORCH_FN(ProfiledType::index_select));
  m.impl_UNBOXED("index_select.dimname", &ProfiledType::index_select_dimname);
  m.impl_UNBOXED("index_select.out", &ProfiledType::index_select_out_out);
  m.impl_UNBOXED("index_select.dimname_out", &ProfiledType::index_select_out_dimname_out);
  m.impl("indices", TORCH_FN(ProfiledType::indices));
  m.impl_UNBOXED("instance_norm", &ProfiledType::instance_norm);
  m.impl("int_repr", TORCH_FN(ProfiledType::int_repr));
  m.impl("inverse", TORCH_FN(ProfiledType::inverse));
  m.impl_UNBOXED("inverse.out", &ProfiledType::inverse_out_out);
  m.impl("irfft", TORCH_FN(ProfiledType::irfft));
  m.impl("is_coalesced", TORCH_FN(ProfiledType::is_coalesced));
  m.impl("is_complex", TORCH_FN(ProfiledType::is_complex));
  m.impl("is_distributed", TORCH_FN(ProfiledType::is_distributed));
  m.impl("is_floating_point", TORCH_FN(ProfiledType::is_floating_point));
  m.impl("is_leaf", TORCH_FN(ProfiledType::is_leaf));
  m.impl("is_nonzero", TORCH_FN(ProfiledType::is_nonzero));
  m.impl("is_pinned", TORCH_FN(ProfiledType::is_pinned));
  m.impl("is_same_size", TORCH_FN(ProfiledType::is_same_size));
  m.impl("is_set_to", TORCH_FN(ProfiledType::is_set_to));
  m.impl("is_signed", TORCH_FN(ProfiledType::is_signed));
  m.impl("is_vulkan_available", TORCH_FN(ProfiledType::is_vulkan_available));
  m.impl("isclose", TORCH_FN(ProfiledType::isclose));
  m.impl("isfinite", TORCH_FN(ProfiledType::isfinite));
  m.impl("isinf", TORCH_FN(ProfiledType::isinf));
  m.impl("isnan", TORCH_FN(ProfiledType::isnan));
  m.impl_UNBOXED("istft", &ProfiledType::istft);
  m.impl("item", TORCH_FN(ProfiledType::item));
  m.impl("kl_div", TORCH_FN(ProfiledType::kl_div));
  m.impl("kl_div_backward", TORCH_FN(ProfiledType::kl_div_backward));
  m.impl("kthvalue", TORCH_FN(ProfiledType::kthvalue));
  m.impl_UNBOXED("kthvalue.dimname", &ProfiledType::kthvalue_dimname);
  m.impl_UNBOXED("kthvalue.values", &ProfiledType::kthvalue_out_values);
  m.impl_UNBOXED("kthvalue.dimname_out", &ProfiledType::kthvalue_out_dimname_out);
  m.impl("l1_loss", TORCH_FN(ProfiledType::l1_loss));
  m.impl("l1_loss_backward", TORCH_FN(ProfiledType::l1_loss_backward));
  m.impl_UNBOXED("l1_loss_backward.grad_input", &ProfiledType::l1_loss_backward_out_grad_input);
  m.impl_UNBOXED("l1_loss.out", &ProfiledType::l1_loss_out_out);
  m.impl_UNBOXED("layer_norm", &ProfiledType::layer_norm);
  m.impl("le.Scalar", TORCH_FN(ProfiledType::le_Scalar));
  m.impl("le.Tensor", TORCH_FN(ProfiledType::le_Tensor));
  m.impl_UNBOXED("le_.Scalar", &ProfiledType::le__Scalar);
  m.impl_UNBOXED("le_.Tensor", &ProfiledType::le__Tensor);
  m.impl_UNBOXED("le.Scalar_out", &ProfiledType::le_out_Scalar_out);
  m.impl_UNBOXED("le.Tensor_out", &ProfiledType::le_out_Tensor_out);
  m.impl("leaky_relu", TORCH_FN(ProfiledType::leaky_relu));
  m.impl_UNBOXED("leaky_relu_", &ProfiledType::leaky_relu_);
  m.impl("leaky_relu_backward", TORCH_FN(ProfiledType::leaky_relu_backward));
  m.impl_UNBOXED("leaky_relu.out", &ProfiledType::leaky_relu_out_out);
  m.impl("lerp.Scalar", TORCH_FN(ProfiledType::lerp_Scalar));
  m.impl("lerp.Tensor", TORCH_FN(ProfiledType::lerp_Tensor));
  m.impl_UNBOXED("lerp_.Scalar", &ProfiledType::lerp__Scalar);
  m.impl_UNBOXED("lerp_.Tensor", &ProfiledType::lerp__Tensor);
  m.impl_UNBOXED("lerp.Scalar_out", &ProfiledType::lerp_out_Scalar_out);
  m.impl_UNBOXED("lerp.Tensor_out", &ProfiledType::lerp_out_Tensor_out);
  m.impl("lgamma", TORCH_FN(ProfiledType::lgamma));
  m.impl_UNBOXED("lgamma_", &ProfiledType::lgamma_);
  m.impl_UNBOXED("lgamma.out", &ProfiledType::lgamma_out_out);
  m.impl_UNBOXED("linear", &ProfiledType::linear);
  m.impl_UNBOXED("linspace", &ProfiledType::linspace);
  m.impl_UNBOXED("linspace.out", &ProfiledType::linspace_out_out);
  m.impl("log", TORCH_FN(ProfiledType::log));
  m.impl("log10", TORCH_FN(ProfiledType::log10));
  m.impl_UNBOXED("log10_", &ProfiledType::log10_);
  m.impl_UNBOXED("log10.out", &ProfiledType::log10_out_out);
  m.impl("log1p", TORCH_FN(ProfiledType::log1p));
  m.impl_UNBOXED("log1p_", &ProfiledType::log1p_);
  m.impl_UNBOXED("log1p.out", &ProfiledType::log1p_out_out);
  m.impl("log2", TORCH_FN(ProfiledType::log2));
  m.impl_UNBOXED("log2_", &ProfiledType::log2_);
  m.impl_UNBOXED("log2.out", &ProfiledType::log2_out_out);
  m.impl_UNBOXED("log_", &ProfiledType::log_);
  m.impl_UNBOXED("log_normal_", &ProfiledType::log_normal_);
  m.impl_UNBOXED("log.out", &ProfiledType::log_out_out);
  m.impl("log_sigmoid", TORCH_FN(ProfiledType::log_sigmoid));
  m.impl("log_sigmoid_backward", TORCH_FN(ProfiledType::log_sigmoid_backward));
  m.impl_UNBOXED("log_sigmoid_backward.grad_input", &ProfiledType::log_sigmoid_backward_out_grad_input);
  m.impl("log_sigmoid_forward", TORCH_FN(ProfiledType::log_sigmoid_forward));
  m.impl_UNBOXED("log_sigmoid_forward.output", &ProfiledType::log_sigmoid_forward_out_output);
  m.impl_UNBOXED("log_sigmoid.out", &ProfiledType::log_sigmoid_out_out);
  m.impl_UNBOXED("log_softmax.int", &ProfiledType::log_softmax_int);
  m.impl_UNBOXED("log_softmax.Dimname", &ProfiledType::log_softmax_Dimname);
  m.impl("logaddexp", TORCH_FN(ProfiledType::logaddexp));
  m.impl("logaddexp2", TORCH_FN(ProfiledType::logaddexp2));
  m.impl_UNBOXED("logaddexp2.out", &ProfiledType::logaddexp2_out_out);
  m.impl_UNBOXED("logaddexp.out", &ProfiledType::logaddexp_out_out);
  m.impl_UNBOXED("logcumsumexp", &ProfiledType::logcumsumexp);
  m.impl_UNBOXED("logcumsumexp.dimname", &ProfiledType::logcumsumexp_dimname);
  m.impl_UNBOXED("logcumsumexp.out", &ProfiledType::logcumsumexp_out_out);
  m.impl_UNBOXED("logcumsumexp.dimname_out", &ProfiledType::logcumsumexp_out_dimname_out);
  m.impl("logdet", TORCH_FN(ProfiledType::logdet));
  m.impl("logical_and", TORCH_FN(ProfiledType::logical_and));
  m.impl_UNBOXED("logical_and_", &ProfiledType::logical_and_);
  m.impl_UNBOXED("logical_and.out", &ProfiledType::logical_and_out_out);
  m.impl("logical_not", TORCH_FN(ProfiledType::logical_not));
  m.impl_UNBOXED("logical_not_", &ProfiledType::logical_not_);
  m.impl_UNBOXED("logical_not.out", &ProfiledType::logical_not_out_out);
  m.impl("logical_or", TORCH_FN(ProfiledType::logical_or));
  m.impl_UNBOXED("logical_or_", &ProfiledType::logical_or_);
  m.impl_UNBOXED("logical_or.out", &ProfiledType::logical_or_out_out);
  m.impl("logical_xor", TORCH_FN(ProfiledType::logical_xor));
  m.impl_UNBOXED("logical_xor_", &ProfiledType::logical_xor_);
  m.impl_UNBOXED("logical_xor.out", &ProfiledType::logical_xor_out_out);
  m.impl_UNBOXED("logspace", &ProfiledType::logspace);
  m.impl_UNBOXED("logspace.out", &ProfiledType::logspace_out_out);
  m.impl("logsumexp", TORCH_FN(ProfiledType::logsumexp));
  m.impl_UNBOXED("logsumexp.names", &ProfiledType::logsumexp_names);
  m.impl_UNBOXED("logsumexp.out", &ProfiledType::logsumexp_out_out);
  m.impl_UNBOXED("logsumexp.names_out", &ProfiledType::logsumexp_out_names_out);
  m.impl("lstm.input", TORCH_FN(ProfiledType::lstm_input));
  m.impl("lstm.data", TORCH_FN(ProfiledType::lstm_data));
  m.impl_UNBOXED("lstm_cell", &ProfiledType::lstm_cell);
  m.impl("lstsq", TORCH_FN(ProfiledType::lstsq));
  m.impl_UNBOXED("lstsq.X", &ProfiledType::lstsq_out_X);
  m.impl("lt.Scalar", TORCH_FN(ProfiledType::lt_Scalar));
  m.impl("lt.Tensor", TORCH_FN(ProfiledType::lt_Tensor));
  m.impl_UNBOXED("lt_.Scalar", &ProfiledType::lt__Scalar);
  m.impl_UNBOXED("lt_.Tensor", &ProfiledType::lt__Tensor);
  m.impl_UNBOXED("lt.Scalar_out", &ProfiledType::lt_out_Scalar_out);
  m.impl_UNBOXED("lt.Tensor_out", &ProfiledType::lt_out_Tensor_out);
  m.impl("lu_solve", TORCH_FN(ProfiledType::lu_solve));
  m.impl_UNBOXED("lu_solve.out", &ProfiledType::lu_solve_out_out);
  m.impl("margin_ranking_loss", TORCH_FN(ProfiledType::margin_ranking_loss));
  m.impl("masked_fill.Scalar", TORCH_FN(ProfiledType::masked_fill_Scalar));
  m.impl("masked_fill.Tensor", TORCH_FN(ProfiledType::masked_fill_Tensor));
  m.impl_UNBOXED("masked_fill_.Scalar", &ProfiledType::masked_fill__Scalar);
  m.impl_UNBOXED("masked_fill_.Tensor", &ProfiledType::masked_fill__Tensor);
  m.impl("masked_scatter", TORCH_FN(ProfiledType::masked_scatter));
  m.impl_UNBOXED("masked_scatter_", &ProfiledType::masked_scatter_);
  m.impl("masked_select", TORCH_FN(ProfiledType::masked_select));
  m.impl_UNBOXED("masked_select.out", &ProfiledType::masked_select_out_out);
  m.impl("matmul", TORCH_FN(ProfiledType::matmul));
  m.impl_UNBOXED("matmul.out", &ProfiledType::matmul_out_out);
  m.impl("matrix_power", TORCH_FN(ProfiledType::matrix_power));
  m.impl("matrix_rank.tol", TORCH_FN(ProfiledType::matrix_rank_tol));
  m.impl("matrix_rank", TORCH_FN(ProfiledType::matrix_rank));
  m.impl("max.dim", TORCH_FN(ProfiledType::max_dim));
  m.impl_UNBOXED("max.names_dim", &ProfiledType::max_names_dim);
  m.impl("max.other", TORCH_FN(ProfiledType::max_other));
  m.impl("max", TORCH_FN(ProfiledType::max));
  m.impl_UNBOXED("max.dim_max", &ProfiledType::max_out_dim_max);
  m.impl_UNBOXED("max.names_dim_max", &ProfiledType::max_out_names_dim_max);
  m.impl_UNBOXED("max.out", &ProfiledType::max_out_out);
  m.impl("max_pool1d", TORCH_FN(ProfiledType::max_pool1d));
  m.impl("max_pool1d_with_indices", TORCH_FN(ProfiledType::max_pool1d_with_indices));
  m.impl("max_pool2d", TORCH_FN(ProfiledType::max_pool2d));
  m.impl("max_pool2d_with_indices", TORCH_FN(ProfiledType::max_pool2d_with_indices));
  m.impl("max_pool2d_with_indices_backward", TORCH_FN(ProfiledType::max_pool2d_with_indices_backward));
  m.impl_UNBOXED("max_pool2d_with_indices_backward.grad_input", &ProfiledType::max_pool2d_with_indices_backward_out_grad_input);
  m.impl_UNBOXED("max_pool2d_with_indices.out", &ProfiledType::max_pool2d_with_indices_out_out);
  m.impl("max_pool3d", TORCH_FN(ProfiledType::max_pool3d));
  m.impl("max_pool3d_with_indices", TORCH_FN(ProfiledType::max_pool3d_with_indices));
  m.impl("max_pool3d_with_indices_backward", TORCH_FN(ProfiledType::max_pool3d_with_indices_backward));
  m.impl_UNBOXED("max_pool3d_with_indices_backward.grad_input", &ProfiledType::max_pool3d_with_indices_backward_out_grad_input);
  m.impl_UNBOXED("max_pool3d_with_indices.out", &ProfiledType::max_pool3d_with_indices_out_out);
  m.impl("max_unpool2d", TORCH_FN(ProfiledType::max_unpool2d));
  m.impl("max_unpool2d_backward", TORCH_FN(ProfiledType::max_unpool2d_backward));
  m.impl_UNBOXED("max_unpool2d_backward.grad_input", &ProfiledType::max_unpool2d_backward_out_grad_input);
  m.impl_UNBOXED("max_unpool2d.out", &ProfiledType::max_unpool2d_out_out);
  m.impl("max_unpool3d", TORCH_FN(ProfiledType::max_unpool3d));
  m.impl("max_unpool3d_backward", TORCH_FN(ProfiledType::max_unpool3d_backward));
  m.impl_UNBOXED("max_unpool3d_backward.grad_input", &ProfiledType::max_unpool3d_backward_out_grad_input);
  m.impl_UNBOXED("max_unpool3d.out", &ProfiledType::max_unpool3d_out_out);
  m.impl("max_values", TORCH_FN(ProfiledType::max_values));
  m.impl_UNBOXED("max_values.names", &ProfiledType::max_values_names);
  m.impl_UNBOXED("mean", &ProfiledType::mean);
  m.impl_UNBOXED("mean.dim", &ProfiledType::mean_dim);
  m.impl_UNBOXED("mean.names_dim", &ProfiledType::mean_names_dim);
  m.impl_UNBOXED("mean.out", &ProfiledType::mean_out_out);
  m.impl_UNBOXED("mean.names_out", &ProfiledType::mean_out_names_out);
  m.impl("median.dim", TORCH_FN(ProfiledType::median_dim));
  m.impl_UNBOXED("median.names_dim", &ProfiledType::median_names_dim);
  m.impl("median", TORCH_FN(ProfiledType::median));
  m.impl_UNBOXED("median.dim_values", &ProfiledType::median_out_dim_values);
  m.impl_UNBOXED("median.names_dim_values", &ProfiledType::median_out_names_dim_values);
  m.impl("meshgrid", TORCH_FN(ProfiledType::meshgrid));
  m.impl("min.dim", TORCH_FN(ProfiledType::min_dim));
  m.impl_UNBOXED("min.names_dim", &ProfiledType::min_names_dim);
  m.impl("min.other", TORCH_FN(ProfiledType::min_other));
  m.impl("min", TORCH_FN(ProfiledType::min));
  m.impl_UNBOXED("min.dim_min", &ProfiledType::min_out_dim_min);
  m.impl_UNBOXED("min.names_dim_min", &ProfiledType::min_out_names_dim_min);
  m.impl_UNBOXED("min.out", &ProfiledType::min_out_out);
  m.impl("min_values", TORCH_FN(ProfiledType::min_values));
  m.impl_UNBOXED("min_values.names", &ProfiledType::min_values_names);
  m.impl_UNBOXED("miopen_batch_norm", &ProfiledType::miopen_batch_norm);
  m.impl_UNBOXED("miopen_batch_norm_backward", &ProfiledType::miopen_batch_norm_backward);
  m.impl_UNBOXED("miopen_convolution", &ProfiledType::miopen_convolution);
  m.impl("miopen_convolution_backward", TORCH_FN(ProfiledType::miopen_convolution_backward));
  m.impl("miopen_convolution_backward_bias", TORCH_FN(ProfiledType::miopen_convolution_backward_bias));
  m.impl("miopen_convolution_backward_input", TORCH_FN(ProfiledType::miopen_convolution_backward_input));
  m.impl("miopen_convolution_backward_weight", TORCH_FN(ProfiledType::miopen_convolution_backward_weight));
  m.impl_UNBOXED("miopen_convolution_transpose", &ProfiledType::miopen_convolution_transpose);
  m.impl("miopen_convolution_transpose_backward", TORCH_FN(ProfiledType::miopen_convolution_transpose_backward));
  m.impl("miopen_convolution_transpose_backward_input", TORCH_FN(ProfiledType::miopen_convolution_transpose_backward_input));
  m.impl("miopen_convolution_transpose_backward_weight", TORCH_FN(ProfiledType::miopen_convolution_transpose_backward_weight));
  m.impl_UNBOXED("miopen_depthwise_convolution", &ProfiledType::miopen_depthwise_convolution);
  m.impl("miopen_depthwise_convolution_backward", TORCH_FN(ProfiledType::miopen_depthwise_convolution_backward));
  m.impl("miopen_depthwise_convolution_backward_input", TORCH_FN(ProfiledType::miopen_depthwise_convolution_backward_input));
  m.impl("miopen_depthwise_convolution_backward_weight", TORCH_FN(ProfiledType::miopen_depthwise_convolution_backward_weight));
  m.impl_UNBOXED("miopen_rnn", &ProfiledType::miopen_rnn);
  m.impl_UNBOXED("miopen_rnn_backward", &ProfiledType::miopen_rnn_backward);
  m.impl("mkldnn_adaptive_avg_pool2d", TORCH_FN(ProfiledType::mkldnn_adaptive_avg_pool2d));
  m.impl_UNBOXED("mkldnn_convolution", &ProfiledType::mkldnn_convolution);
  m.impl("mkldnn_convolution_backward", TORCH_FN(ProfiledType::mkldnn_convolution_backward));
  m.impl("mkldnn_convolution_backward_input", TORCH_FN(ProfiledType::mkldnn_convolution_backward_input));
  m.impl("mkldnn_convolution_backward_weights", TORCH_FN(ProfiledType::mkldnn_convolution_backward_weights));
  m.impl_UNBOXED("mkldnn_linear", &ProfiledType::mkldnn_linear);
  m.impl("mkldnn_max_pool2d", TORCH_FN(ProfiledType::mkldnn_max_pool2d));
  m.impl("mkldnn_reorder_conv2d_weight", TORCH_FN(ProfiledType::mkldnn_reorder_conv2d_weight));
  m.impl("mm", TORCH_FN(ProfiledType::mm));
  m.impl_UNBOXED("mm.out", &ProfiledType::mm_out_out);
  m.impl("mode", TORCH_FN(ProfiledType::mode));
  m.impl_UNBOXED("mode.dimname", &ProfiledType::mode_dimname);
  m.impl_UNBOXED("mode.values", &ProfiledType::mode_out_values);
  m.impl_UNBOXED("mode.dimname_out", &ProfiledType::mode_out_dimname_out);
  m.impl("mse_loss", TORCH_FN(ProfiledType::mse_loss));
  m.impl("mse_loss_backward", TORCH_FN(ProfiledType::mse_loss_backward));
  m.impl_UNBOXED("mse_loss_backward.grad_input", &ProfiledType::mse_loss_backward_out_grad_input);
  m.impl_UNBOXED("mse_loss.out", &ProfiledType::mse_loss_out_out);
  m.impl("mul.Tensor", TORCH_FN(ProfiledType::mul_Tensor));
  m.impl("mul.Scalar", TORCH_FN(ProfiledType::mul_Scalar));
  m.impl_UNBOXED("mul_.Tensor", &ProfiledType::mul__Tensor);
  m.impl_UNBOXED("mul_.Scalar", &ProfiledType::mul__Scalar);
  m.impl_UNBOXED("mul.out", &ProfiledType::mul_out_out);
  m.impl_UNBOXED("multi_margin_loss", &ProfiledType::multi_margin_loss);
  m.impl_UNBOXED("multi_margin_loss_backward", &ProfiledType::multi_margin_loss_backward);
  m.impl_UNBOXED("multi_margin_loss_backward.grad_input", &ProfiledType::multi_margin_loss_backward_out_grad_input);
  m.impl_UNBOXED("multi_margin_loss.out", &ProfiledType::multi_margin_loss_out_out);
  m.impl("multilabel_margin_loss", TORCH_FN(ProfiledType::multilabel_margin_loss));
  m.impl("multilabel_margin_loss_backward", TORCH_FN(ProfiledType::multilabel_margin_loss_backward));
  m.impl_UNBOXED("multilabel_margin_loss_backward.grad_input", &ProfiledType::multilabel_margin_loss_backward_out_grad_input);
  m.impl("multilabel_margin_loss_forward", TORCH_FN(ProfiledType::multilabel_margin_loss_forward));
  m.impl_UNBOXED("multilabel_margin_loss_forward.output", &ProfiledType::multilabel_margin_loss_forward_out_output);
  m.impl_UNBOXED("multilabel_margin_loss.out", &ProfiledType::multilabel_margin_loss_out_out);
  m.impl_UNBOXED("multinomial", &ProfiledType::multinomial);
  m.impl_UNBOXED("multinomial.out", &ProfiledType::multinomial_out_out);
  m.impl("mv", TORCH_FN(ProfiledType::mv));
  m.impl_UNBOXED("mv.out", &ProfiledType::mv_out_out);
  m.impl("mvlgamma", TORCH_FN(ProfiledType::mvlgamma));
  m.impl_UNBOXED("mvlgamma_", &ProfiledType::mvlgamma_);
  m.impl("narrow", TORCH_FN(ProfiledType::narrow));
  m.impl("narrow.Tensor", TORCH_FN(ProfiledType::narrow_Tensor));
  m.impl("narrow_copy", TORCH_FN(ProfiledType::narrow_copy));
  m.impl_UNBOXED("native_batch_norm", &ProfiledType::native_batch_norm);
  m.impl_UNBOXED("native_batch_norm_backward", &ProfiledType::native_batch_norm_backward);
  m.impl_UNBOXED("native_batch_norm.out", &ProfiledType::native_batch_norm_out_out);
  m.impl_UNBOXED("native_group_norm", &ProfiledType::native_group_norm);
  m.impl_UNBOXED("native_group_norm_backward", &ProfiledType::native_group_norm_backward);
  m.impl_UNBOXED("native_layer_norm", &ProfiledType::native_layer_norm);
  m.impl_UNBOXED("native_layer_norm_backward", &ProfiledType::native_layer_norm_backward);
  m.impl("native_norm", TORCH_FN(ProfiledType::native_norm));
  m.impl("ne.Scalar", TORCH_FN(ProfiledType::ne_Scalar));
  m.impl("ne.Tensor", TORCH_FN(ProfiledType::ne_Tensor));
  m.impl_UNBOXED("ne_.Scalar", &ProfiledType::ne__Scalar);
  m.impl_UNBOXED("ne_.Tensor", &ProfiledType::ne__Tensor);
  m.impl_UNBOXED("ne.Scalar_out", &ProfiledType::ne_out_Scalar_out);
  m.impl_UNBOXED("ne.Tensor_out", &ProfiledType::ne_out_Tensor_out);
  m.impl("neg", TORCH_FN(ProfiledType::neg));
  m.impl_UNBOXED("neg_", &ProfiledType::neg_);
  m.impl_UNBOXED("neg.out", &ProfiledType::neg_out_out);
  m.impl_UNBOXED("new_empty", &ProfiledType::new_empty);
  m.impl_UNBOXED("new_full", &ProfiledType::new_full);
  m.impl_UNBOXED("new_zeros", &ProfiledType::new_zeros);
  m.impl_UNBOXED("nll_loss", &ProfiledType::nll_loss);
  m.impl_UNBOXED("nll_loss2d", &ProfiledType::nll_loss2d);
  m.impl_UNBOXED("nll_loss2d_backward", &ProfiledType::nll_loss2d_backward);
  m.impl_UNBOXED("nll_loss2d_backward.grad_input", &ProfiledType::nll_loss2d_backward_out_grad_input);
  m.impl_UNBOXED("nll_loss2d_forward", &ProfiledType::nll_loss2d_forward);
  m.impl_UNBOXED("nll_loss2d_forward.output", &ProfiledType::nll_loss2d_forward_out_output);
  m.impl_UNBOXED("nll_loss2d.out", &ProfiledType::nll_loss2d_out_out);
  m.impl_UNBOXED("nll_loss_backward", &ProfiledType::nll_loss_backward);
  m.impl_UNBOXED("nll_loss_backward.grad_input", &ProfiledType::nll_loss_backward_out_grad_input);
  m.impl_UNBOXED("nll_loss_forward", &ProfiledType::nll_loss_forward);
  m.impl_UNBOXED("nll_loss_forward.output", &ProfiledType::nll_loss_forward_out_output);
  m.impl_UNBOXED("nll_loss.out", &ProfiledType::nll_loss_out_out);
  m.impl("nonzero", TORCH_FN(ProfiledType::nonzero));
  m.impl("nonzero_numpy", TORCH_FN(ProfiledType::nonzero_numpy));
  m.impl_UNBOXED("nonzero.out", &ProfiledType::nonzero_out_out);
  m.impl_UNBOXED("norm.ScalarOpt_dtype", &ProfiledType::norm_ScalarOpt_dtype);
  m.impl("norm.Scalar", TORCH_FN(ProfiledType::norm_Scalar));
  m.impl_UNBOXED("norm.ScalarOpt_dim_dtype", &ProfiledType::norm_ScalarOpt_dim_dtype);
  m.impl("norm.ScalarOpt_dim", TORCH_FN(ProfiledType::norm_ScalarOpt_dim));
  m.impl_UNBOXED("norm.names_ScalarOpt_dim_dtype", &ProfiledType::norm_names_ScalarOpt_dim_dtype);
  m.impl_UNBOXED("norm.names_ScalarOpt_dim", &ProfiledType::norm_names_ScalarOpt_dim);
  m.impl("norm_except_dim", TORCH_FN(ProfiledType::norm_except_dim));
  m.impl_UNBOXED("norm.dtype_out", &ProfiledType::norm_out_dtype_out);
  m.impl_UNBOXED("norm.out", &ProfiledType::norm_out_out);
  m.impl_UNBOXED("norm.names_dtype_out", &ProfiledType::norm_out_names_dtype_out);
  m.impl_UNBOXED("norm.names_out", &ProfiledType::norm_out_names_out);
  m.impl_UNBOXED("normal.Tensor_float", &ProfiledType::normal_Tensor_float);
  m.impl_UNBOXED("normal.float_Tensor", &ProfiledType::normal_float_Tensor);
  m.impl_UNBOXED("normal.Tensor_Tensor", &ProfiledType::normal_Tensor_Tensor);
  m.impl_UNBOXED("normal.float_float", &ProfiledType::normal_float_float);
  m.impl_UNBOXED("normal_", &ProfiledType::normal_);
  m.impl_UNBOXED("normal.Tensor_float_out", &ProfiledType::normal_out_Tensor_float_out);
  m.impl_UNBOXED("normal.float_Tensor_out", &ProfiledType::normal_out_float_Tensor_out);
  m.impl_UNBOXED("normal.Tensor_Tensor_out", &ProfiledType::normal_out_Tensor_Tensor_out);
  m.impl_UNBOXED("normal.float_float_out", &ProfiledType::normal_out_float_float_out);
  m.impl("nuclear_norm", TORCH_FN(ProfiledType::nuclear_norm));
  m.impl("nuclear_norm.dim", TORCH_FN(ProfiledType::nuclear_norm_dim));
  m.impl_UNBOXED("nuclear_norm.out", &ProfiledType::nuclear_norm_out_out);
  m.impl_UNBOXED("nuclear_norm.dim_out", &ProfiledType::nuclear_norm_out_dim_out);
  m.impl("numpy_T", TORCH_FN(ProfiledType::numpy_T));
  m.impl("one_hot", TORCH_FN(ProfiledType::one_hot));
  m.impl_UNBOXED("ones.names", &ProfiledType::ones_names);
  m.impl_UNBOXED("ones", &ProfiledType::ones);
  m.impl_UNBOXED("ones_like", &ProfiledType::ones_like);
  m.impl_UNBOXED("ones.out", &ProfiledType::ones_out_out);
  m.impl("orgqr", TORCH_FN(ProfiledType::orgqr));
  m.impl_UNBOXED("orgqr.out", &ProfiledType::orgqr_out_out);
  m.impl("ormqr", TORCH_FN(ProfiledType::ormqr));
  m.impl_UNBOXED("ormqr.out", &ProfiledType::ormqr_out_out);
  m.impl("output_nr", TORCH_FN(ProfiledType::output_nr));
  m.impl("pairwise_distance", TORCH_FN(ProfiledType::pairwise_distance));
  m.impl("pdist", TORCH_FN(ProfiledType::pdist));
  m.impl("permute", TORCH_FN(ProfiledType::permute));
  m.impl("pin_memory", TORCH_FN(ProfiledType::pin_memory));
  m.impl("pinverse", TORCH_FN(ProfiledType::pinverse));
  m.impl("pixel_shuffle", TORCH_FN(ProfiledType::pixel_shuffle));
  m.impl_UNBOXED("poisson", &ProfiledType::poisson);
  m.impl("poisson_nll_loss", TORCH_FN(ProfiledType::poisson_nll_loss));
  m.impl("polygamma", TORCH_FN(ProfiledType::polygamma));
  m.impl_UNBOXED("polygamma_", &ProfiledType::polygamma_);
  m.impl_UNBOXED("polygamma.out", &ProfiledType::polygamma_out_out);
  m.impl("pow.Tensor_Scalar", TORCH_FN(ProfiledType::pow_Tensor_Scalar));
  m.impl("pow.Tensor_Tensor", TORCH_FN(ProfiledType::pow_Tensor_Tensor));
  m.impl("pow.Scalar", TORCH_FN(ProfiledType::pow_Scalar));
  m.impl_UNBOXED("pow_.Scalar", &ProfiledType::pow__Scalar);
  m.impl_UNBOXED("pow_.Tensor", &ProfiledType::pow__Tensor);
  m.impl_UNBOXED("pow.Tensor_Scalar_out", &ProfiledType::pow_out_Tensor_Scalar_out);
  m.impl_UNBOXED("pow.Tensor_Tensor_out", &ProfiledType::pow_out_Tensor_Tensor_out);
  m.impl_UNBOXED("pow.Scalar_out", &ProfiledType::pow_out_Scalar_out);
  m.impl("prelu", TORCH_FN(ProfiledType::prelu));
  m.impl("prelu_backward", TORCH_FN(ProfiledType::prelu_backward));
  m.impl_UNBOXED("prod", &ProfiledType::prod);
  m.impl_UNBOXED("prod.dim_int", &ProfiledType::prod_dim_int);
  m.impl_UNBOXED("prod.dim_Dimname", &ProfiledType::prod_dim_Dimname);
  m.impl_UNBOXED("prod.int_out", &ProfiledType::prod_out_int_out);
  m.impl_UNBOXED("prod.Dimname_out", &ProfiledType::prod_out_Dimname_out);
  m.impl_UNBOXED("promote_types", &ProfiledType::promote_types);
  m.impl_UNBOXED("put_", &ProfiledType::put_);
  m.impl("q_per_channel_axis", TORCH_FN(ProfiledType::q_per_channel_axis));
  m.impl("q_per_channel_scales", TORCH_FN(ProfiledType::q_per_channel_scales));
  m.impl("q_per_channel_zero_points", TORCH_FN(ProfiledType::q_per_channel_zero_points));
  m.impl("q_scale", TORCH_FN(ProfiledType::q_scale));
  m.impl("q_zero_point", TORCH_FN(ProfiledType::q_zero_point));
  m.impl("qr", TORCH_FN(ProfiledType::qr));
  m.impl_UNBOXED("qr.Q", &ProfiledType::qr_out_Q);
  m.impl("qscheme", TORCH_FN(ProfiledType::qscheme));
  m.impl_UNBOXED("quantize_per_channel", &ProfiledType::quantize_per_channel);
  m.impl_UNBOXED("quantize_per_tensor", &ProfiledType::quantize_per_tensor);
  m.impl_UNBOXED("quantize_per_tensor.tensors", &ProfiledType::quantize_per_tensor_tensors);
  m.impl_UNBOXED("quantized_batch_norm", &ProfiledType::quantized_batch_norm);
  m.impl("quantized_gru_cell", TORCH_FN(ProfiledType::quantized_gru_cell));
  m.impl("quantized_lstm_cell", TORCH_FN(ProfiledType::quantized_lstm_cell));
  m.impl("quantized_max_pool2d", TORCH_FN(ProfiledType::quantized_max_pool2d));
  m.impl("quantized_rnn_relu_cell", TORCH_FN(ProfiledType::quantized_rnn_relu_cell));
  m.impl("quantized_rnn_tanh_cell", TORCH_FN(ProfiledType::quantized_rnn_tanh_cell));
  m.impl("rad2deg", TORCH_FN(ProfiledType::rad2deg));
  m.impl_UNBOXED("rad2deg_", &ProfiledType::rad2deg_);
  m.impl_UNBOXED("rad2deg.out", &ProfiledType::rad2deg_out_out);
  m.impl_UNBOXED("rand.names", &ProfiledType::rand_names);
  m.impl_UNBOXED("rand.generator_with_names", &ProfiledType::rand_generator_with_names);
  m.impl_UNBOXED("rand", &ProfiledType::rand);
  m.impl_UNBOXED("rand.generator", &ProfiledType::rand_generator);
  m.impl_UNBOXED("rand_like", &ProfiledType::rand_like);
  m.impl_UNBOXED("rand.out", &ProfiledType::rand_out_out);
  m.impl_UNBOXED("rand.generator_out", &ProfiledType::rand_out_generator_out);
  m.impl_UNBOXED("randint", &ProfiledType::randint);
  m.impl_UNBOXED("randint.generator", &ProfiledType::randint_generator);
  m.impl_UNBOXED("randint.low", &ProfiledType::randint_low);
  m.impl_UNBOXED("randint.low_generator", &ProfiledType::randint_low_generator);
  m.impl_UNBOXED("randint_like", &ProfiledType::randint_like);
  m.impl_UNBOXED("randint_like.low_dtype", &ProfiledType::randint_like_low_dtype);
  m.impl_UNBOXED("randint.out", &ProfiledType::randint_out_out);
  m.impl_UNBOXED("randint.generator_out", &ProfiledType::randint_out_generator_out);
  m.impl_UNBOXED("randint.low_out", &ProfiledType::randint_out_low_out);
  m.impl_UNBOXED("randint.low_generator_out", &ProfiledType::randint_out_low_generator_out);
  m.impl_UNBOXED("randn", &ProfiledType::randn);
  m.impl_UNBOXED("randn.generator", &ProfiledType::randn_generator);
  m.impl_UNBOXED("randn.names", &ProfiledType::randn_names);
  m.impl_UNBOXED("randn.generator_with_names", &ProfiledType::randn_generator_with_names);
  m.impl_UNBOXED("randn_like", &ProfiledType::randn_like);
  m.impl_UNBOXED("randn.out", &ProfiledType::randn_out_out);
  m.impl_UNBOXED("randn.generator_out", &ProfiledType::randn_out_generator_out);
  m.impl_UNBOXED("random_.from", &ProfiledType::random__from);
  m.impl_UNBOXED("random_.to", &ProfiledType::random__to);
  m.impl_UNBOXED("random_", &ProfiledType::random_);
  m.impl_UNBOXED("randperm", &ProfiledType::randperm);
  m.impl_UNBOXED("randperm.generator", &ProfiledType::randperm_generator);
  m.impl_UNBOXED("randperm.out", &ProfiledType::randperm_out_out);
  m.impl_UNBOXED("randperm.generator_out", &ProfiledType::randperm_out_generator_out);
  m.impl_UNBOXED("range.step", &ProfiledType::range_step);
  m.impl_UNBOXED("range", &ProfiledType::range);
  m.impl_UNBOXED("range.out", &ProfiledType::range_out_out);
  m.impl("real", TORCH_FN(ProfiledType::real));
  m.impl("reciprocal", TORCH_FN(ProfiledType::reciprocal));
  m.impl_UNBOXED("reciprocal_", &ProfiledType::reciprocal_);
  m.impl_UNBOXED("reciprocal.out", &ProfiledType::reciprocal_out_out);
  m.impl_UNBOXED("refine_names", &ProfiledType::refine_names);
  m.impl("reflection_pad1d", TORCH_FN(ProfiledType::reflection_pad1d));
  m.impl("reflection_pad1d_backward", TORCH_FN(ProfiledType::reflection_pad1d_backward));
  m.impl_UNBOXED("reflection_pad1d_backward.grad_input", &ProfiledType::reflection_pad1d_backward_out_grad_input);
  m.impl_UNBOXED("reflection_pad1d.out", &ProfiledType::reflection_pad1d_out_out);
  m.impl("reflection_pad2d", TORCH_FN(ProfiledType::reflection_pad2d));
  m.impl("reflection_pad2d_backward", TORCH_FN(ProfiledType::reflection_pad2d_backward));
  m.impl_UNBOXED("reflection_pad2d_backward.grad_input", &ProfiledType::reflection_pad2d_backward_out_grad_input);
  m.impl_UNBOXED("reflection_pad2d.out", &ProfiledType::reflection_pad2d_out_out);
  m.impl("relu", TORCH_FN(ProfiledType::relu));
  m.impl_UNBOXED("relu_", &ProfiledType::relu_);
  m.impl("remainder.Scalar", TORCH_FN(ProfiledType::remainder_Scalar));
  m.impl("remainder.Tensor", TORCH_FN(ProfiledType::remainder_Tensor));
  m.impl_UNBOXED("remainder_.Scalar", &ProfiledType::remainder__Scalar);
  m.impl_UNBOXED("remainder_.Tensor", &ProfiledType::remainder__Tensor);
  m.impl_UNBOXED("remainder.Scalar_out", &ProfiledType::remainder_out_Scalar_out);
  m.impl_UNBOXED("remainder.Tensor_out", &ProfiledType::remainder_out_Tensor_out);
  m.impl_UNBOXED("rename", &ProfiledType::rename);
  m.impl_UNBOXED("rename_", &ProfiledType::rename_);
  m.impl("renorm", TORCH_FN(ProfiledType::renorm));
  m.impl_UNBOXED("renorm_", &ProfiledType::renorm_);
  m.impl_UNBOXED("renorm.out", &ProfiledType::renorm_out_out);
  m.impl("repeat", TORCH_FN(ProfiledType::repeat));
  m.impl("repeat_interleave.Tensor", TORCH_FN(ProfiledType::repeat_interleave_Tensor));
  m.impl("repeat_interleave.self_Tensor", TORCH_FN(ProfiledType::repeat_interleave_self_Tensor));
  m.impl("repeat_interleave.self_int", TORCH_FN(ProfiledType::repeat_interleave_self_int));
  m.impl("replication_pad1d", TORCH_FN(ProfiledType::replication_pad1d));
  m.impl("replication_pad1d_backward", TORCH_FN(ProfiledType::replication_pad1d_backward));
  m.impl_UNBOXED("replication_pad1d_backward.grad_input", &ProfiledType::replication_pad1d_backward_out_grad_input);
  m.impl_UNBOXED("replication_pad1d.out", &ProfiledType::replication_pad1d_out_out);
  m.impl("replication_pad2d", TORCH_FN(ProfiledType::replication_pad2d));
  m.impl("replication_pad2d_backward", TORCH_FN(ProfiledType::replication_pad2d_backward));
  m.impl_UNBOXED("replication_pad2d_backward.grad_input", &ProfiledType::replication_pad2d_backward_out_grad_input);
  m.impl_UNBOXED("replication_pad2d.out", &ProfiledType::replication_pad2d_out_out);
  m.impl("replication_pad3d", TORCH_FN(ProfiledType::replication_pad3d));
  m.impl("replication_pad3d_backward", TORCH_FN(ProfiledType::replication_pad3d_backward));
  m.impl_UNBOXED("replication_pad3d_backward.grad_input", &ProfiledType::replication_pad3d_backward_out_grad_input);
  m.impl_UNBOXED("replication_pad3d.out", &ProfiledType::replication_pad3d_out_out);
  m.impl_UNBOXED("requires_grad_", &ProfiledType::requires_grad_);
  m.impl("reshape", TORCH_FN(ProfiledType::reshape));
  m.impl("reshape_as", TORCH_FN(ProfiledType::reshape_as));
  m.impl_UNBOXED("resize_", &ProfiledType::resize_);
  m.impl_UNBOXED("resize_as_", &ProfiledType::resize_as_);
  m.impl_UNBOXED("result_type.Tensor", &ProfiledType::result_type_Tensor);
  m.impl_UNBOXED("result_type.Scalar", &ProfiledType::result_type_Scalar);
  m.impl_UNBOXED("result_type.Scalar_Tensor", &ProfiledType::result_type_Scalar_Tensor);
  m.impl_UNBOXED("result_type.Scalar_Scalar", &ProfiledType::result_type_Scalar_Scalar);
  m.impl("retain_grad", TORCH_FN(ProfiledType::retain_grad));
  m.impl("rfft", TORCH_FN(ProfiledType::rfft));
  m.impl("rnn_relu.input", TORCH_FN(ProfiledType::rnn_relu_input));
  m.impl("rnn_relu.data", TORCH_FN(ProfiledType::rnn_relu_data));
  m.impl_UNBOXED("rnn_relu_cell", &ProfiledType::rnn_relu_cell);
  m.impl("rnn_tanh.input", TORCH_FN(ProfiledType::rnn_tanh_input));
  m.impl("rnn_tanh.data", TORCH_FN(ProfiledType::rnn_tanh_data));
  m.impl_UNBOXED("rnn_tanh_cell", &ProfiledType::rnn_tanh_cell);
  m.impl("roll", TORCH_FN(ProfiledType::roll));
  m.impl("rot90", TORCH_FN(ProfiledType::rot90));
  m.impl("round", TORCH_FN(ProfiledType::round));
  m.impl_UNBOXED("round_", &ProfiledType::round_);
  m.impl_UNBOXED("round.out", &ProfiledType::round_out_out);
  m.impl_UNBOXED("rrelu", &ProfiledType::rrelu);
  m.impl_UNBOXED("rrelu_", &ProfiledType::rrelu_);
  m.impl_UNBOXED("rrelu_with_noise", &ProfiledType::rrelu_with_noise);
  m.impl_UNBOXED("rrelu_with_noise_", &ProfiledType::rrelu_with_noise_);
  m.impl("rrelu_with_noise_backward", TORCH_FN(ProfiledType::rrelu_with_noise_backward));
  m.impl_UNBOXED("rrelu_with_noise.out", &ProfiledType::rrelu_with_noise_out_out);
  m.impl("rsqrt", TORCH_FN(ProfiledType::rsqrt));
  m.impl_UNBOXED("rsqrt_", &ProfiledType::rsqrt_);
  m.impl_UNBOXED("rsqrt.out", &ProfiledType::rsqrt_out_out);
  m.impl("rsub.Tensor", TORCH_FN(ProfiledType::rsub_Tensor));
  m.impl("rsub.Scalar", TORCH_FN(ProfiledType::rsub_Scalar));
  m.impl_UNBOXED("scalar_tensor", &ProfiledType::scalar_tensor);
  m.impl("scatter.src", TORCH_FN(ProfiledType::scatter_src));
  m.impl("scatter.value", TORCH_FN(ProfiledType::scatter_value));
  m.impl_UNBOXED("scatter.dimname_src", &ProfiledType::scatter_dimname_src);
  m.impl_UNBOXED("scatter.dimname_value", &ProfiledType::scatter_dimname_value);
  m.impl_UNBOXED("scatter_.src", &ProfiledType::scatter__src);
  m.impl_UNBOXED("scatter_.value", &ProfiledType::scatter__value);
  m.impl("scatter_add", TORCH_FN(ProfiledType::scatter_add));
  m.impl_UNBOXED("scatter_add.dimname", &ProfiledType::scatter_add_dimname);
  m.impl_UNBOXED("scatter_add_", &ProfiledType::scatter_add_);
  m.impl("searchsorted.Tensor", TORCH_FN(ProfiledType::searchsorted_Tensor));
  m.impl("searchsorted.Scalar", TORCH_FN(ProfiledType::searchsorted_Scalar));
  m.impl_UNBOXED("searchsorted.Tensor_out", &ProfiledType::searchsorted_out_Tensor_out);
  m.impl_UNBOXED("select.Dimname", &ProfiledType::select_Dimname);
  m.impl("select.int", TORCH_FN(ProfiledType::select_int));
  m.impl("selu", TORCH_FN(ProfiledType::selu));
  m.impl_UNBOXED("selu_", &ProfiledType::selu_);
  m.impl_UNBOXED("set_.source_Storage", &ProfiledType::set__source_Storage);
  m.impl_UNBOXED("set_.source_Storage_storage_offset", &ProfiledType::set__source_Storage_storage_offset);
  m.impl_UNBOXED("set_.source_Tensor", &ProfiledType::set__source_Tensor);
  m.impl_UNBOXED("set_", &ProfiledType::set_);
  m.impl("set_data", TORCH_FN(ProfiledType::set_data));
  m.impl_UNBOXED("set_quantizer_", &ProfiledType::set_quantizer_);
  m.impl("sigmoid", TORCH_FN(ProfiledType::sigmoid));
  m.impl_UNBOXED("sigmoid_", &ProfiledType::sigmoid_);
  m.impl("sigmoid_backward", TORCH_FN(ProfiledType::sigmoid_backward));
  m.impl_UNBOXED("sigmoid_backward.grad_input", &ProfiledType::sigmoid_backward_out_grad_input);
  m.impl_UNBOXED("sigmoid.out", &ProfiledType::sigmoid_out_out);
  m.impl("sign", TORCH_FN(ProfiledType::sign));
  m.impl_UNBOXED("sign_", &ProfiledType::sign_);
  m.impl_UNBOXED("sign.out", &ProfiledType::sign_out_out);
  m.impl("sin", TORCH_FN(ProfiledType::sin));
  m.impl_UNBOXED("sin_", &ProfiledType::sin_);
  m.impl_UNBOXED("sin.out", &ProfiledType::sin_out_out);
  m.impl("sinh", TORCH_FN(ProfiledType::sinh));
  m.impl_UNBOXED("sinh_", &ProfiledType::sinh_);
  m.impl_UNBOXED("sinh.out", &ProfiledType::sinh_out_out);
  m.impl("size.int", TORCH_FN(ProfiledType::size_int));
  m.impl_UNBOXED("size.Dimname", &ProfiledType::size_Dimname);
  m.impl("slice.Tensor", TORCH_FN(ProfiledType::slice_Tensor));
  m.impl("slogdet", TORCH_FN(ProfiledType::slogdet));
  m.impl_UNBOXED("slow_conv3d", &ProfiledType::slow_conv3d);
  m.impl("slow_conv3d_backward.output_mask", TORCH_FN(ProfiledType::slow_conv3d_backward_output_mask));
  m.impl_UNBOXED("slow_conv3d_backward.grad_input", &ProfiledType::slow_conv3d_backward_out_grad_input);
  m.impl_UNBOXED("slow_conv3d_forward", &ProfiledType::slow_conv3d_forward);
  m.impl_UNBOXED("slow_conv3d_forward.output", &ProfiledType::slow_conv3d_forward_out_output);
  m.impl_UNBOXED("slow_conv3d.out", &ProfiledType::slow_conv3d_out_out);
  m.impl_UNBOXED("slow_conv_dilated2d", &ProfiledType::slow_conv_dilated2d);
  m.impl("slow_conv_dilated2d_backward", TORCH_FN(ProfiledType::slow_conv_dilated2d_backward));
  m.impl_UNBOXED("slow_conv_dilated3d", &ProfiledType::slow_conv_dilated3d);
  m.impl("slow_conv_dilated3d_backward", TORCH_FN(ProfiledType::slow_conv_dilated3d_backward));
  m.impl_UNBOXED("slow_conv_transpose2d", &ProfiledType::slow_conv_transpose2d);
  m.impl("slow_conv_transpose2d_backward.output_mask", TORCH_FN(ProfiledType::slow_conv_transpose2d_backward_output_mask));
  m.impl_UNBOXED("slow_conv_transpose2d_backward.grad_output", &ProfiledType::slow_conv_transpose2d_backward_out_grad_output);
  m.impl_UNBOXED("slow_conv_transpose2d.out", &ProfiledType::slow_conv_transpose2d_out_out);
  m.impl_UNBOXED("slow_conv_transpose3d", &ProfiledType::slow_conv_transpose3d);
  m.impl("slow_conv_transpose3d_backward.output_mask", TORCH_FN(ProfiledType::slow_conv_transpose3d_backward_output_mask));
  m.impl_UNBOXED("slow_conv_transpose3d_backward.grad_output", &ProfiledType::slow_conv_transpose3d_backward_out_grad_output);
  m.impl_UNBOXED("slow_conv_transpose3d.out", &ProfiledType::slow_conv_transpose3d_out_out);
  m.impl("smm", TORCH_FN(ProfiledType::smm));
  m.impl("smooth_l1_loss", TORCH_FN(ProfiledType::smooth_l1_loss));
  m.impl("smooth_l1_loss_backward", TORCH_FN(ProfiledType::smooth_l1_loss_backward));
  m.impl_UNBOXED("smooth_l1_loss_backward.grad_input", &ProfiledType::smooth_l1_loss_backward_out_grad_input);
  m.impl_UNBOXED("smooth_l1_loss.out", &ProfiledType::smooth_l1_loss_out_out);
  m.impl("soft_margin_loss", TORCH_FN(ProfiledType::soft_margin_loss));
  m.impl("soft_margin_loss_backward", TORCH_FN(ProfiledType::soft_margin_loss_backward));
  m.impl_UNBOXED("soft_margin_loss_backward.grad_input", &ProfiledType::soft_margin_loss_backward_out_grad_input);
  m.impl_UNBOXED("soft_margin_loss.out", &ProfiledType::soft_margin_loss_out_out);
  m.impl_UNBOXED("softmax.int", &ProfiledType::softmax_int);
  m.impl_UNBOXED("softmax.Dimname", &ProfiledType::softmax_Dimname);
  m.impl("softplus", TORCH_FN(ProfiledType::softplus));
  m.impl("softplus_backward", TORCH_FN(ProfiledType::softplus_backward));
  m.impl_UNBOXED("softplus_backward.grad_input", &ProfiledType::softplus_backward_out_grad_input);
  m.impl_UNBOXED("softplus.out", &ProfiledType::softplus_out_out);
  m.impl("softshrink", TORCH_FN(ProfiledType::softshrink));
  m.impl("softshrink_backward", TORCH_FN(ProfiledType::softshrink_backward));
  m.impl_UNBOXED("softshrink_backward.grad_input", &ProfiledType::softshrink_backward_out_grad_input);
  m.impl_UNBOXED("softshrink.out", &ProfiledType::softshrink_out_out);
  m.impl("solve", TORCH_FN(ProfiledType::solve));
  m.impl_UNBOXED("solve.solution", &ProfiledType::solve_out_solution);
  m.impl("sort", TORCH_FN(ProfiledType::sort));
  m.impl_UNBOXED("sort.dimname", &ProfiledType::sort_dimname);
  m.impl_UNBOXED("sort.values", &ProfiledType::sort_out_values);
  m.impl_UNBOXED("sort.dimname_values", &ProfiledType::sort_out_dimname_values);
  m.impl_UNBOXED("sparse_coo_tensor.size", &ProfiledType::sparse_coo_tensor_size);
  m.impl_UNBOXED("sparse_coo_tensor.indices", &ProfiledType::sparse_coo_tensor_indices);
  m.impl_UNBOXED("sparse_coo_tensor.indices_size", &ProfiledType::sparse_coo_tensor_indices_size);
  m.impl("sparse_dim", TORCH_FN(ProfiledType::sparse_dim));
  m.impl("sparse_mask", TORCH_FN(ProfiledType::sparse_mask));
  m.impl_UNBOXED("sparse_resize_", &ProfiledType::sparse_resize_);
  m.impl_UNBOXED("sparse_resize_and_clear_", &ProfiledType::sparse_resize_and_clear_);
  m.impl("split.Tensor", TORCH_FN(ProfiledType::split_Tensor));
  m.impl("split_with_sizes", TORCH_FN(ProfiledType::split_with_sizes));
  m.impl("sqrt", TORCH_FN(ProfiledType::sqrt));
  m.impl_UNBOXED("sqrt_", &ProfiledType::sqrt_);
  m.impl_UNBOXED("sqrt.out", &ProfiledType::sqrt_out_out);
  m.impl("square", TORCH_FN(ProfiledType::square));
  m.impl_UNBOXED("square_", &ProfiledType::square_);
  m.impl("squeeze", TORCH_FN(ProfiledType::squeeze));
  m.impl("squeeze.dim", TORCH_FN(ProfiledType::squeeze_dim));
  m.impl_UNBOXED("squeeze.dimname", &ProfiledType::squeeze_dimname);
  m.impl_UNBOXED("squeeze_", &ProfiledType::squeeze_);
  m.impl_UNBOXED("squeeze_.dim", &ProfiledType::squeeze__dim);
  m.impl_UNBOXED("squeeze_.dimname", &ProfiledType::squeeze__dimname);
  m.impl("sspaddmm", TORCH_FN(ProfiledType::sspaddmm));
  m.impl_UNBOXED("sspaddmm.out", &ProfiledType::sspaddmm_out_out);
  m.impl("stack", TORCH_FN(ProfiledType::stack));
  m.impl_UNBOXED("stack.out", &ProfiledType::stack_out_out);
  m.impl("std", TORCH_FN(ProfiledType::std));
  m.impl("std.dim", TORCH_FN(ProfiledType::std_dim));
  m.impl_UNBOXED("std.names_dim", &ProfiledType::std_names_dim);
  m.impl("std_mean", TORCH_FN(ProfiledType::std_mean));
  m.impl("std_mean.dim", TORCH_FN(ProfiledType::std_mean_dim));
  m.impl_UNBOXED("std_mean.names_dim", &ProfiledType::std_mean_names_dim);
  m.impl_UNBOXED("std.out", &ProfiledType::std_out_out);
  m.impl_UNBOXED("std.names_out", &ProfiledType::std_out_names_out);
  m.impl_UNBOXED("stft", &ProfiledType::stft);
  m.impl("stride.int", TORCH_FN(ProfiledType::stride_int));
  m.impl_UNBOXED("stride.Dimname", &ProfiledType::stride_Dimname);
  m.impl("sub.Tensor", TORCH_FN(ProfiledType::sub_Tensor));
  m.impl("sub.Scalar", TORCH_FN(ProfiledType::sub_Scalar));
  m.impl_UNBOXED("sub_.Tensor", &ProfiledType::sub__Tensor);
  m.impl_UNBOXED("sub_.Scalar", &ProfiledType::sub__Scalar);
  m.impl_UNBOXED("sub.out", &ProfiledType::sub_out_out);
  m.impl_UNBOXED("sum", &ProfiledType::sum);
  m.impl_UNBOXED("sum.dim_IntList", &ProfiledType::sum_dim_IntList);
  m.impl_UNBOXED("sum.dim_DimnameList", &ProfiledType::sum_dim_DimnameList);
  m.impl_UNBOXED("sum.IntList_out", &ProfiledType::sum_out_IntList_out);
  m.impl_UNBOXED("sum.DimnameList_out", &ProfiledType::sum_out_DimnameList_out);
  m.impl("sum_to_size", TORCH_FN(ProfiledType::sum_to_size));
  m.impl("svd", TORCH_FN(ProfiledType::svd));
  m.impl_UNBOXED("svd.U", &ProfiledType::svd_out_U);
  m.impl("symeig", TORCH_FN(ProfiledType::symeig));
  m.impl_UNBOXED("symeig.e", &ProfiledType::symeig_out_e);
  m.impl("t", TORCH_FN(ProfiledType::t));
  m.impl_UNBOXED("t_", &ProfiledType::t_);
  m.impl("take", TORCH_FN(ProfiledType::take));
  m.impl_UNBOXED("take.out", &ProfiledType::take_out_out);
  m.impl("tan", TORCH_FN(ProfiledType::tan));
  m.impl_UNBOXED("tan_", &ProfiledType::tan_);
  m.impl_UNBOXED("tan.out", &ProfiledType::tan_out_out);
  m.impl("tanh", TORCH_FN(ProfiledType::tanh));
  m.impl_UNBOXED("tanh_", &ProfiledType::tanh_);
  m.impl("tanh_backward", TORCH_FN(ProfiledType::tanh_backward));
  m.impl_UNBOXED("tanh_backward.grad_input", &ProfiledType::tanh_backward_out_grad_input);
  m.impl_UNBOXED("tanh.out", &ProfiledType::tanh_out_out);
  m.impl("tensordot", TORCH_FN(ProfiledType::tensordot));
  m.impl_UNBOXED("thnn_conv2d", &ProfiledType::thnn_conv2d);
  m.impl("thnn_conv2d_backward.output_mask", TORCH_FN(ProfiledType::thnn_conv2d_backward_output_mask));
  m.impl_UNBOXED("thnn_conv2d_backward.grad_input", &ProfiledType::thnn_conv2d_backward_out_grad_input);
  m.impl_UNBOXED("thnn_conv2d_forward", &ProfiledType::thnn_conv2d_forward);
  m.impl_UNBOXED("thnn_conv2d_forward.output", &ProfiledType::thnn_conv2d_forward_out_output);
  m.impl_UNBOXED("thnn_conv2d.out", &ProfiledType::thnn_conv2d_out_out);
  m.impl_UNBOXED("thnn_conv_depthwise2d", &ProfiledType::thnn_conv_depthwise2d);
  m.impl("thnn_conv_depthwise2d_backward.output_mask", TORCH_FN(ProfiledType::thnn_conv_depthwise2d_backward_output_mask));
  m.impl_UNBOXED("thnn_conv_depthwise2d_backward.grad_input", &ProfiledType::thnn_conv_depthwise2d_backward_out_grad_input);
  m.impl_UNBOXED("thnn_conv_depthwise2d_forward", &ProfiledType::thnn_conv_depthwise2d_forward);
  m.impl_UNBOXED("thnn_conv_depthwise2d_forward.out", &ProfiledType::thnn_conv_depthwise2d_forward_out_out);
  m.impl_UNBOXED("thnn_conv_depthwise2d.out", &ProfiledType::thnn_conv_depthwise2d_out_out);
  m.impl("threshold", TORCH_FN(ProfiledType::threshold));
  m.impl_UNBOXED("threshold_", &ProfiledType::threshold_);
  m.impl("threshold_backward", TORCH_FN(ProfiledType::threshold_backward));
  m.impl_UNBOXED("threshold.out", &ProfiledType::threshold_out_out);
  m.impl_UNBOXED("to.dtype_layout", &ProfiledType::to_dtype_layout);
  m.impl_UNBOXED("to.device", &ProfiledType::to_device);
  m.impl_UNBOXED("to.dtype", &ProfiledType::to_dtype);
  m.impl_UNBOXED("to.other", &ProfiledType::to_other);
  m.impl("to_dense", TORCH_FN(ProfiledType::to_dense));
  m.impl("to_dense_backward", TORCH_FN(ProfiledType::to_dense_backward));
  m.impl("to_mkldnn", TORCH_FN(ProfiledType::to_mkldnn));
  m.impl("to_mkldnn_backward", TORCH_FN(ProfiledType::to_mkldnn_backward));
  m.impl("to_sparse.sparse_dim", TORCH_FN(ProfiledType::to_sparse_sparse_dim));
  m.impl("to_sparse", TORCH_FN(ProfiledType::to_sparse));
  m.impl("topk", TORCH_FN(ProfiledType::topk));
  m.impl_UNBOXED("topk.values", &ProfiledType::topk_out_values);
  m.impl("trace", TORCH_FN(ProfiledType::trace));
  m.impl("transpose.int", TORCH_FN(ProfiledType::transpose_int));
  m.impl_UNBOXED("transpose.Dimname", &ProfiledType::transpose_Dimname);
  m.impl_UNBOXED("transpose_", &ProfiledType::transpose_);
  m.impl("trapz.x", TORCH_FN(ProfiledType::trapz_x));
  m.impl("trapz.dx", TORCH_FN(ProfiledType::trapz_dx));
  m.impl("triangular_solve", TORCH_FN(ProfiledType::triangular_solve));
  m.impl_UNBOXED("triangular_solve.X", &ProfiledType::triangular_solve_out_X);
  m.impl("tril", TORCH_FN(ProfiledType::tril));
  m.impl_UNBOXED("tril_", &ProfiledType::tril_);
  m.impl_UNBOXED("tril_indices", &ProfiledType::tril_indices);
  m.impl_UNBOXED("tril.out", &ProfiledType::tril_out_out);
  m.impl("triplet_margin_loss", TORCH_FN(ProfiledType::triplet_margin_loss));
  m.impl("triu", TORCH_FN(ProfiledType::triu));
  m.impl_UNBOXED("triu_", &ProfiledType::triu_);
  m.impl_UNBOXED("triu_indices", &ProfiledType::triu_indices);
  m.impl_UNBOXED("triu.out", &ProfiledType::triu_out_out);
  m.impl("true_divide.Tensor", TORCH_FN(ProfiledType::true_divide_Tensor));
  m.impl("true_divide.Scalar", TORCH_FN(ProfiledType::true_divide_Scalar));
  m.impl_UNBOXED("true_divide_.Tensor", &ProfiledType::true_divide__Tensor);
  m.impl_UNBOXED("true_divide_.Scalar", &ProfiledType::true_divide__Scalar);
  m.impl_UNBOXED("true_divide.out", &ProfiledType::true_divide_out_out);
  m.impl("trunc", TORCH_FN(ProfiledType::trunc));
  m.impl_UNBOXED("trunc_", &ProfiledType::trunc_);
  m.impl_UNBOXED("trunc.out", &ProfiledType::trunc_out_out);
  m.impl("type_as", TORCH_FN(ProfiledType::type_as));
  m.impl("unbind.int", TORCH_FN(ProfiledType::unbind_int));
  m.impl_UNBOXED("unbind.Dimname", &ProfiledType::unbind_Dimname);
  m.impl_UNBOXED("unflatten.Dimname", &ProfiledType::unflatten_Dimname);
  m.impl_UNBOXED("unflatten.int", &ProfiledType::unflatten_int);
  m.impl("unfold", TORCH_FN(ProfiledType::unfold));
  m.impl_UNBOXED("unfold_backward", &ProfiledType::unfold_backward);
  m.impl_UNBOXED("uniform_", &ProfiledType::uniform_);
  m.impl("unique_consecutive", TORCH_FN(ProfiledType::unique_consecutive));
  m.impl("unique_dim", TORCH_FN(ProfiledType::unique_dim));
  m.impl("unique_dim_consecutive", TORCH_FN(ProfiledType::unique_dim_consecutive));
  m.impl("unsqueeze", TORCH_FN(ProfiledType::unsqueeze));
  m.impl_UNBOXED("unsqueeze_", &ProfiledType::unsqueeze_);
  m.impl("upsample_bicubic2d", TORCH_FN(ProfiledType::upsample_bicubic2d));
  m.impl("upsample_bicubic2d_backward", TORCH_FN(ProfiledType::upsample_bicubic2d_backward));
  m.impl_UNBOXED("upsample_bicubic2d_backward.grad_input", &ProfiledType::upsample_bicubic2d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_bicubic2d.out", &ProfiledType::upsample_bicubic2d_out_out);
  m.impl("upsample_bilinear2d", TORCH_FN(ProfiledType::upsample_bilinear2d));
  m.impl("upsample_bilinear2d_backward", TORCH_FN(ProfiledType::upsample_bilinear2d_backward));
  m.impl_UNBOXED("upsample_bilinear2d_backward.grad_input", &ProfiledType::upsample_bilinear2d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_bilinear2d.out", &ProfiledType::upsample_bilinear2d_out_out);
  m.impl("upsample_linear1d", TORCH_FN(ProfiledType::upsample_linear1d));
  m.impl("upsample_linear1d_backward", TORCH_FN(ProfiledType::upsample_linear1d_backward));
  m.impl_UNBOXED("upsample_linear1d_backward.grad_input", &ProfiledType::upsample_linear1d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_linear1d.out", &ProfiledType::upsample_linear1d_out_out);
  m.impl("upsample_nearest1d", TORCH_FN(ProfiledType::upsample_nearest1d));
  m.impl("upsample_nearest1d_backward", TORCH_FN(ProfiledType::upsample_nearest1d_backward));
  m.impl_UNBOXED("upsample_nearest1d_backward.grad_input", &ProfiledType::upsample_nearest1d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_nearest1d.out", &ProfiledType::upsample_nearest1d_out_out);
  m.impl("upsample_nearest2d", TORCH_FN(ProfiledType::upsample_nearest2d));
  m.impl("upsample_nearest2d_backward", TORCH_FN(ProfiledType::upsample_nearest2d_backward));
  m.impl_UNBOXED("upsample_nearest2d_backward.grad_input", &ProfiledType::upsample_nearest2d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_nearest2d.out", &ProfiledType::upsample_nearest2d_out_out);
  m.impl("upsample_nearest3d", TORCH_FN(ProfiledType::upsample_nearest3d));
  m.impl("upsample_nearest3d_backward", TORCH_FN(ProfiledType::upsample_nearest3d_backward));
  m.impl_UNBOXED("upsample_nearest3d_backward.grad_input", &ProfiledType::upsample_nearest3d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_nearest3d.out", &ProfiledType::upsample_nearest3d_out_out);
  m.impl("upsample_trilinear3d", TORCH_FN(ProfiledType::upsample_trilinear3d));
  m.impl("upsample_trilinear3d_backward", TORCH_FN(ProfiledType::upsample_trilinear3d_backward));
  m.impl_UNBOXED("upsample_trilinear3d_backward.grad_input", &ProfiledType::upsample_trilinear3d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_trilinear3d.out", &ProfiledType::upsample_trilinear3d_out_out);
  m.impl("values", TORCH_FN(ProfiledType::values));
  m.impl("vander", TORCH_FN(ProfiledType::vander));
  m.impl("var", TORCH_FN(ProfiledType::var));
  m.impl("var.dim", TORCH_FN(ProfiledType::var_dim));
  m.impl_UNBOXED("var.names_dim", &ProfiledType::var_names_dim);
  m.impl("var_mean", TORCH_FN(ProfiledType::var_mean));
  m.impl("var_mean.dim", TORCH_FN(ProfiledType::var_mean_dim));
  m.impl_UNBOXED("var_mean.names_dim", &ProfiledType::var_mean_names_dim);
  m.impl_UNBOXED("var.out", &ProfiledType::var_out_out);
  m.impl_UNBOXED("var.names_out", &ProfiledType::var_out_names_out);
  m.impl("view", TORCH_FN(ProfiledType::view));
  m.impl("view_as", TORCH_FN(ProfiledType::view_as));
  m.impl("view_as_complex", TORCH_FN(ProfiledType::view_as_complex));
  m.impl("view_as_real", TORCH_FN(ProfiledType::view_as_real));
  m.impl("where.self", TORCH_FN(ProfiledType::where_self));
  m.impl("where", TORCH_FN(ProfiledType::where));
  m.impl_UNBOXED("zero_", &ProfiledType::zero_);
  m.impl_UNBOXED("zeros.names", &ProfiledType::zeros_names);
  m.impl_UNBOXED("zeros", &ProfiledType::zeros);
  m.impl_UNBOXED("zeros_like", &ProfiledType::zeros_like);
  m.impl_UNBOXED("zeros.out", &ProfiledType::zeros_out_out);;
}

}  // namespace

} // namespace torch
