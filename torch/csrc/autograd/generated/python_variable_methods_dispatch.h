#pragma once

// generated from tools/autograd/templates/python_variable_methods_dispatch.h

#include <ATen/ATen.h>
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/auto_gpu.h"

// Contains inline wrappers around ATen functions that release the GIL and
// switch to the correct CUDA device.

namespace torch { namespace autograd {

using at::Tensor;
using at::Scalar;
using at::TensorList;
using at::IntList;
using at::Generator;
using at::SparseTensor;
using at::Storage;

inline Tensor dispatch___and__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__and__(other);
}
inline Tensor dispatch___and__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__and__(other);
}
inline Tensor & dispatch___iand__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__iand__(other);
}
inline Tensor & dispatch___iand__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__iand__(other);
}
inline Tensor & dispatch___ilshift__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__ilshift__(other);
}
inline Tensor & dispatch___ilshift__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__ilshift__(other);
}
inline Tensor & dispatch___ior__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__ior__(other);
}
inline Tensor & dispatch___ior__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__ior__(other);
}
inline Tensor & dispatch___irshift__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__irshift__(other);
}
inline Tensor & dispatch___irshift__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__irshift__(other);
}
inline Tensor & dispatch___ixor__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__ixor__(other);
}
inline Tensor & dispatch___ixor__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__ixor__(other);
}
inline Tensor dispatch___lshift__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__lshift__(other);
}
inline Tensor dispatch___lshift__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__lshift__(other);
}
inline Tensor dispatch___or__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__or__(other);
}
inline Tensor dispatch___or__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__or__(other);
}
inline Tensor dispatch___rshift__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__rshift__(other);
}
inline Tensor dispatch___rshift__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__rshift__(other);
}
inline Tensor dispatch___xor__(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__xor__(other);
}
inline Tensor dispatch___xor__(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.__xor__(other);
}
inline Tensor dispatch__addmv(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._addmv(mat, vec, beta, alpha);
}
inline Tensor & dispatch__addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._addmv_(mat, vec, beta, alpha);
}
inline Tensor dispatch__addr(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._addr(vec1, vec2, beta, alpha);
}
inline Tensor & dispatch__addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._addr_(vec1, vec2, beta, alpha);
}
inline Tensor & dispatch__copy_ignoring_overlaps_(Tensor & self, const Tensor & src) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._copy_ignoring_overlaps_(src);
}
inline std::tuple<Tensor,Tensor,Tensor,Tensor> dispatch__det_with_svd(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._det_with_svd();
}
inline int64_t dispatch__dimI(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._dimI();
}
inline int64_t dispatch__dimV(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._dimV();
}
inline Tensor dispatch__dot(Tensor & self, const Tensor & tensor) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._dot(tensor);
}
inline Tensor dispatch__ger(Tensor & self, const Tensor & vec2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._ger(vec2);
}
inline Tensor dispatch__indices(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._indices();
}
inline Tensor dispatch__mm(Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._mm(mat2);
}
inline Tensor dispatch__mv(Tensor & self, const Tensor & vec) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._mv(vec);
}
inline int64_t dispatch__nnz(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._nnz();
}
inline Tensor dispatch__s_where(const Tensor & condition, Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(condition);
  return self._s_where(condition, other);
}
inline Tensor dispatch__sparse_mask(Tensor & self, SparseTensor mask) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._sparse_mask(mask);
}
inline Tensor dispatch__standard_gamma(Tensor & self, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._standard_gamma(generator);
}
inline Tensor dispatch__standard_gamma_grad(Tensor & self, const Tensor & output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._standard_gamma_grad(output);
}
inline Tensor dispatch__values(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self._values();
}
inline Tensor dispatch_abs(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.abs();
}
inline Tensor & dispatch_abs_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.abs_();
}
inline Tensor dispatch_acos(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.acos();
}
inline Tensor & dispatch_acos_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.acos_();
}
inline Tensor dispatch_add(Tensor & self, Scalar alpha, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.add(other, alpha);
}
inline Tensor dispatch_add(Tensor & self, Scalar other, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.add(other, alpha);
}
inline Tensor dispatch_add(Tensor & self, const Tensor & other, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.add(other, alpha);
}
inline Tensor & dispatch_add_(Tensor & self, Scalar alpha, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.add_(other, alpha);
}
inline Tensor & dispatch_add_(Tensor & self, Scalar other, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.add_(other, alpha);
}
inline Tensor & dispatch_add_(Tensor & self, const Tensor & other, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.add_(other, alpha);
}
inline Tensor dispatch_addbmm(Scalar beta, Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addbmm(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_addbmm(Scalar beta, Tensor & self, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addbmm(batch1, batch2, beta, 1);
}
inline Tensor dispatch_addbmm(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addbmm(batch1, batch2, beta, alpha);
}
inline Tensor & dispatch_addbmm_(Scalar beta, Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addbmm_(batch1, batch2, beta, alpha);
}
inline Tensor & dispatch_addbmm_(Scalar beta, Tensor & self, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addbmm_(batch1, batch2, beta, 1);
}
inline Tensor & dispatch_addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addbmm_(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_addcdiv(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addcdiv(tensor1, tensor2, value);
}
inline Tensor dispatch_addcdiv(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addcdiv(tensor1, tensor2, value);
}
inline Tensor & dispatch_addcdiv_(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addcdiv_(tensor1, tensor2, value);
}
inline Tensor & dispatch_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addcdiv_(tensor1, tensor2, value);
}
inline Tensor dispatch_addcmul(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addcmul(tensor1, tensor2, value);
}
inline Tensor dispatch_addcmul(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addcmul(tensor1, tensor2, value);
}
inline Tensor & dispatch_addcmul_(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addcmul_(tensor1, tensor2, value);
}
inline Tensor & dispatch_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addcmul_(tensor1, tensor2, value);
}
inline Tensor dispatch_addmm(Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmm(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_addmm(Scalar beta, Tensor & self, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmm(mat1, mat2, beta, 1);
}
inline Tensor dispatch_addmm(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmm(mat1, mat2, beta, alpha);
}
inline Tensor & dispatch_addmm_(Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmm_(mat1, mat2, beta, alpha);
}
inline Tensor & dispatch_addmm_(Scalar beta, Tensor & self, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmm_(mat1, mat2, beta, 1);
}
inline Tensor & dispatch_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmm_(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_addmv(Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmv(mat, vec, beta, alpha);
}
inline Tensor dispatch_addmv(Scalar beta, Tensor & self, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmv(mat, vec, beta, 1);
}
inline Tensor dispatch_addmv(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmv(mat, vec, beta, alpha);
}
inline Tensor & dispatch_addmv_(Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmv_(mat, vec, beta, alpha);
}
inline Tensor & dispatch_addmv_(Scalar beta, Tensor & self, const Tensor & mat, const Tensor & vec) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmv_(mat, vec, beta, 1);
}
inline Tensor & dispatch_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addmv_(mat, vec, beta, alpha);
}
inline Tensor dispatch_addr(Scalar beta, Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addr(vec1, vec2, beta, alpha);
}
inline Tensor dispatch_addr(Scalar beta, Tensor & self, const Tensor & vec1, const Tensor & vec2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addr(vec1, vec2, beta, 1);
}
inline Tensor dispatch_addr(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addr(vec1, vec2, beta, alpha);
}
inline Tensor & dispatch_addr_(Scalar beta, Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addr_(vec1, vec2, beta, alpha);
}
inline Tensor & dispatch_addr_(Scalar beta, Tensor & self, const Tensor & vec1, const Tensor & vec2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addr_(vec1, vec2, beta, 1);
}
inline Tensor & dispatch_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.addr_(vec1, vec2, beta, alpha);
}
inline Tensor dispatch_all(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.all();
}
inline bool dispatch_allclose(Tensor & self, const Tensor & other, double rtol, double atol) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.allclose(other, rtol, atol);
}
inline Tensor dispatch_any(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.any();
}
inline Tensor dispatch_as_strided(Tensor & self, IntList size, IntList stride, int64_t storage_offset) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.as_strided(size, stride, storage_offset);
}
inline Tensor & dispatch_as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.as_strided_(size, stride, storage_offset);
}
inline Tensor dispatch_asin(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.asin();
}
inline Tensor & dispatch_asin_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.asin_();
}
inline Tensor dispatch_atan(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.atan();
}
inline Tensor dispatch_atan2(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.atan2(other);
}
inline Tensor & dispatch_atan2_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.atan2_(other);
}
inline Tensor & dispatch_atan_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.atan_();
}
inline Tensor dispatch_baddbmm(Scalar beta, Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.baddbmm(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_baddbmm(Scalar beta, Tensor & self, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.baddbmm(batch1, batch2, beta, 1);
}
inline Tensor dispatch_baddbmm(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.baddbmm(batch1, batch2, beta, alpha);
}
inline Tensor & dispatch_baddbmm_(Scalar beta, Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.baddbmm_(batch1, batch2, beta, alpha);
}
inline Tensor & dispatch_baddbmm_(Scalar beta, Tensor & self, const Tensor & batch1, const Tensor & batch2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.baddbmm_(batch1, batch2, beta, 1);
}
inline Tensor & dispatch_baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.baddbmm_(batch1, batch2, beta, alpha);
}
inline Tensor dispatch_bernoulli(Tensor & self, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.bernoulli(generator);
}
inline Tensor & dispatch_bernoulli_(Tensor & self, const Tensor & p, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.bernoulli_(p, generator);
}
inline Tensor & dispatch_bernoulli_(Tensor & self, double p, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.bernoulli_(p, generator);
}
inline Tensor dispatch_bmm(Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.bmm(mat2);
}
inline std::tuple<Tensor,Tensor> dispatch_btrifact(Tensor & self, bool pivot) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.btrifact(pivot);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_btrifact_with_info(Tensor & self, bool pivot) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.btrifact_with_info(pivot);
}
inline Tensor dispatch_btrisolve(Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.btrisolve(LU_data, LU_pivots);
}
inline Tensor & dispatch_cauchy_(Tensor & self, double median, double sigma, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cauchy_(median, sigma, generator);
}
inline Tensor dispatch_ceil(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ceil();
}
inline Tensor & dispatch_ceil_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ceil_();
}
inline std::vector<Tensor> dispatch_chunk(Tensor & self, int64_t chunks, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.chunk(chunks, dim);
}
inline Tensor dispatch_clone(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clone();
}
inline Tensor dispatch_coalesce(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.coalesce();
}
inline Tensor dispatch_conv_tbc(Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.conv_tbc(weight, bias, pad);
}
inline Tensor dispatch_cos(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cos();
}
inline Tensor & dispatch_cos_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cos_();
}
inline Tensor dispatch_cosh(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cosh();
}
inline Tensor & dispatch_cosh_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cosh_();
}
inline Tensor dispatch_cross(Tensor & self, const Tensor & other, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cross(other, dim);
}
inline Tensor dispatch_cumprod(Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cumprod(dim);
}
inline Tensor dispatch_cumsum(Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.cumsum(dim);
}
inline void* dispatch_data_ptr(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.data_ptr();
}
inline Tensor dispatch_det(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.det();
}
inline Tensor dispatch_diag(Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.diag(diagonal);
}
inline Tensor dispatch_digamma(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.digamma();
}
inline Tensor & dispatch_digamma_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.digamma_();
}
inline Tensor dispatch_dist(Tensor & self, const Tensor & other, Scalar p) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.dist(other, p);
}
inline Tensor dispatch_div(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.div(other);
}
inline Tensor dispatch_div(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.div(other);
}
inline Tensor & dispatch_div_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.div_(other);
}
inline Tensor & dispatch_div_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.div_(other);
}
inline Tensor dispatch_dot(Tensor & self, const Tensor & tensor) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.dot(tensor);
}
inline std::tuple<Tensor,Tensor> dispatch_eig(Tensor & self, bool eigenvectors) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.eig(eigenvectors);
}
inline Tensor dispatch_eq(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.eq(other);
}
inline Tensor dispatch_eq(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.eq(other);
}
inline Tensor & dispatch_eq_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.eq_(other);
}
inline Tensor & dispatch_eq_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.eq_(other);
}
inline bool dispatch_equal(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.equal(other);
}
inline Tensor dispatch_erf(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.erf();
}
inline Tensor & dispatch_erf_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.erf_();
}
inline Tensor dispatch_erfinv(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.erfinv();
}
inline Tensor & dispatch_erfinv_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.erfinv_();
}
inline Tensor dispatch_exp(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.exp();
}
inline Tensor & dispatch_exp_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.exp_();
}
inline Tensor dispatch_expand(Tensor & self, IntList size) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.expand(size);
}
inline Tensor dispatch_expand_as(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.expand_as(other);
}
inline Tensor dispatch_expm1(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.expm1();
}
inline Tensor & dispatch_expm1_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.expm1_();
}
inline Tensor & dispatch_exponential_(Tensor & self, double lambd, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.exponential_(lambd, generator);
}
inline Tensor & dispatch_fill_(Tensor & self, Scalar value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.fill_(value);
}
inline Tensor & dispatch_fill_(Tensor & self, const Tensor & value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.fill_(value);
}
inline Tensor dispatch_floor(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.floor();
}
inline Tensor & dispatch_floor_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.floor_();
}
inline Tensor dispatch_fmod(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.fmod(other);
}
inline Tensor dispatch_fmod(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.fmod(other);
}
inline Tensor & dispatch_fmod_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.fmod_(other);
}
inline Tensor & dispatch_fmod_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.fmod_(other);
}
inline Tensor dispatch_frac(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.frac();
}
inline Tensor & dispatch_frac_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.frac_();
}
inline Tensor dispatch_gather(Tensor & self, int64_t dim, const Tensor & index) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.gather(dim, index);
}
inline Tensor dispatch_ge(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ge(other);
}
inline Tensor dispatch_ge(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ge(other);
}
inline Tensor & dispatch_ge_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ge_(other);
}
inline Tensor & dispatch_ge_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ge_(other);
}
inline std::tuple<Tensor,Tensor> dispatch_gels(Tensor & self, const Tensor & A) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.gels(A);
}
inline Tensor & dispatch_geometric_(Tensor & self, double p, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.geometric_(p, generator);
}
inline std::tuple<Tensor,Tensor> dispatch_geqrf(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.geqrf();
}
inline Tensor dispatch_ger(Tensor & self, const Tensor & vec2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ger(vec2);
}
inline std::tuple<Tensor,Tensor> dispatch_gesv(Tensor & self, const Tensor & A) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.gesv(A);
}
inline int64_t dispatch_get_device(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.get_device();
}
inline Tensor dispatch_gt(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.gt(other);
}
inline Tensor dispatch_gt(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.gt(other);
}
inline Tensor & dispatch_gt_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.gt_(other);
}
inline Tensor & dispatch_gt_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.gt_(other);
}
inline Tensor dispatch_histc(Tensor & self, int64_t bins, Scalar min, Scalar max) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.histc(bins, min, max);
}
inline Tensor dispatch_index(Tensor & self, TensorList indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.index(indices);
}
inline Tensor & dispatch_index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.index_add_(dim, index, source);
}
inline Tensor & dispatch_index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.index_copy_(dim, index, source);
}
inline Tensor & dispatch_index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.index_fill_(dim, index, value);
}
inline Tensor & dispatch_index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.index_fill_(dim, index, value);
}
inline Tensor & dispatch_index_put_(Tensor & self, TensorList indices, const Tensor & values) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.index_put_(indices, values);
}
inline Tensor dispatch_index_select(Tensor & self, int64_t dim, const Tensor & index) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.index_select(dim, index);
}
inline Tensor dispatch_inverse(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.inverse();
}
inline bool dispatch_is_coalesced(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.is_coalesced();
}
inline bool dispatch_is_contiguous(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.is_contiguous();
}
inline bool dispatch_is_distributed(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.is_distributed();
}
inline bool dispatch_is_floating_point(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.is_floating_point();
}
inline bool dispatch_is_nonzero(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.is_nonzero();
}
inline bool dispatch_is_same_size(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.is_same_size(other);
}
inline bool dispatch_is_set_to(Tensor & self, const Tensor & tensor) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.is_set_to(tensor);
}
inline bool dispatch_is_signed(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.is_signed();
}
inline std::tuple<Tensor,Tensor> dispatch_kthvalue(Tensor & self, int64_t k, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.kthvalue(k, dim, keepdim);
}
inline Tensor dispatch_le(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.le(other);
}
inline Tensor dispatch_le(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.le(other);
}
inline Tensor & dispatch_le_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.le_(other);
}
inline Tensor & dispatch_le_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.le_(other);
}
inline Tensor dispatch_lerp(Tensor & self, const Tensor & end, Scalar weight) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.lerp(end, weight);
}
inline Tensor & dispatch_lerp_(Tensor & self, const Tensor & end, Scalar weight) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.lerp_(end, weight);
}
inline Tensor dispatch_lgamma(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.lgamma();
}
inline Tensor & dispatch_lgamma_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.lgamma_();
}
inline Tensor dispatch_log(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.log();
}
inline Tensor dispatch_log1p(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.log1p();
}
inline Tensor & dispatch_log1p_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.log1p_();
}
inline Tensor & dispatch_log_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.log_();
}
inline Tensor & dispatch_log_normal_(Tensor & self, double mean, double std, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.log_normal_(mean, std, generator);
}
inline Tensor dispatch_lt(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.lt(other);
}
inline Tensor dispatch_lt(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.lt(other);
}
inline Tensor & dispatch_lt_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.lt_(other);
}
inline Tensor & dispatch_lt_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.lt_(other);
}
inline Tensor & dispatch_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.masked_fill_(mask, value);
}
inline Tensor & dispatch_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.masked_fill_(mask, value);
}
inline Tensor & dispatch_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.masked_scatter_(mask, source);
}
inline Tensor dispatch_masked_select(Tensor & self, const Tensor & mask) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.masked_select(mask);
}
inline Tensor dispatch_matmul(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.matmul(other);
}
inline Tensor dispatch_max(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.max();
}
inline Tensor dispatch_max(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.max(other);
}
inline std::tuple<Tensor,Tensor> dispatch_max(Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.max(dim, keepdim);
}
inline Tensor dispatch_mean(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mean();
}
inline Tensor dispatch_mean(Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mean(dim, keepdim);
}
inline Tensor dispatch_median(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.median();
}
inline std::tuple<Tensor,Tensor> dispatch_median(Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.median(dim, keepdim);
}
inline Tensor dispatch_min(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.min();
}
inline Tensor dispatch_min(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.min(other);
}
inline std::tuple<Tensor,Tensor> dispatch_min(Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.min(dim, keepdim);
}
inline Tensor dispatch_mm(Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mm(mat2);
}
inline std::tuple<Tensor,Tensor> dispatch_mode(Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mode(dim, keepdim);
}
inline Tensor dispatch_mul(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mul(other);
}
inline Tensor dispatch_mul(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mul(other);
}
inline Tensor & dispatch_mul_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mul_(other);
}
inline Tensor & dispatch_mul_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mul_(other);
}
inline Tensor dispatch_multinomial(Tensor & self, int64_t num_samples, bool replacement, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.multinomial(num_samples, replacement, generator);
}
inline Tensor dispatch_mv(Tensor & self, const Tensor & vec) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.mv(vec);
}
inline Tensor dispatch_narrow(Tensor & self, int64_t dim, int64_t start, int64_t length) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.narrow(dim, start, length);
}
inline Tensor dispatch_ne(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ne(other);
}
inline Tensor dispatch_ne(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ne(other);
}
inline Tensor & dispatch_ne_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ne_(other);
}
inline Tensor & dispatch_ne_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ne_(other);
}
inline Tensor dispatch_neg(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.neg();
}
inline Tensor & dispatch_neg_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.neg_();
}
inline Tensor dispatch_nonzero(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.nonzero();
}
inline Tensor dispatch_norm(Tensor & self, Scalar p) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.norm(p);
}
inline Tensor dispatch_norm(Tensor & self, Scalar p, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.norm(p, dim, keepdim);
}
inline Tensor & dispatch_normal_(Tensor & self, double mean, double std, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.normal_(mean, std, generator);
}
inline int64_t dispatch_numel(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.numel();
}
inline Tensor dispatch_orgqr(Tensor & self, const Tensor & input2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.orgqr(input2);
}
inline Tensor dispatch_ormqr(Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.ormqr(input2, input3, left, transpose);
}
inline Tensor dispatch_permute(Tensor & self, IntList dims) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.permute(dims);
}
inline Tensor dispatch_pin_memory(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.pin_memory();
}
inline Tensor dispatch_polygamma(int64_t n, Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.polygamma(n);
}
inline Tensor & dispatch_polygamma_(Tensor & self, int64_t n) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.polygamma_(n);
}
inline Tensor dispatch_potrf(Tensor & self, bool upper) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.potrf(upper);
}
inline Tensor dispatch_potri(Tensor & self, bool upper) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.potri(upper);
}
inline Tensor dispatch_potrs(Tensor & self, const Tensor & input2, bool upper) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.potrs(input2, upper);
}
inline Tensor dispatch_pow(Tensor & self, Scalar exponent) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.pow(exponent);
}
inline Tensor dispatch_pow(Tensor & self, const Tensor & exponent) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.pow(exponent);
}
inline Tensor & dispatch_pow_(Tensor & self, Scalar exponent) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.pow_(exponent);
}
inline Tensor & dispatch_pow_(Tensor & self, const Tensor & exponent) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.pow_(exponent);
}
inline Tensor dispatch_prod(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.prod();
}
inline Tensor dispatch_prod(Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.prod(dim, keepdim);
}
inline std::tuple<Tensor,Tensor> dispatch_pstrf(Tensor & self, bool upper, Scalar tol) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.pstrf(upper, tol);
}
inline Tensor & dispatch_put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.put_(index, source, accumulate);
}
inline std::tuple<Tensor,Tensor> dispatch_qr(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.qr();
}
inline Tensor & dispatch_random_(Tensor & self, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.random_(generator);
}
inline Tensor & dispatch_random_(Tensor & self, int64_t from, int64_t to, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.random_(from, to, generator);
}
inline Tensor & dispatch_random_(Tensor & self, int64_t to, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.random_(to, generator);
}
inline Tensor dispatch_reciprocal(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.reciprocal();
}
inline Tensor & dispatch_reciprocal_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.reciprocal_();
}
inline Tensor dispatch_remainder(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.remainder(other);
}
inline Tensor dispatch_remainder(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.remainder(other);
}
inline Tensor & dispatch_remainder_(Tensor & self, Scalar other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.remainder_(other);
}
inline Tensor & dispatch_remainder_(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.remainder_(other);
}
inline Tensor dispatch_renorm(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.renorm(p, dim, maxnorm);
}
inline Tensor & dispatch_renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.renorm_(p, dim, maxnorm);
}
inline Tensor dispatch_repeat(Tensor & self, IntList repeats) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.repeat(repeats);
}
inline Tensor & dispatch_reshape_(Tensor & self, IntList size, IntList stride) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.reshape_(size, stride);
}
inline Tensor & dispatch_resize_(Tensor & self, IntList size) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.resize_(size);
}
inline Tensor & dispatch_resize_as_(Tensor & self, const Tensor & the_template) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.resize_as_(the_template);
}
inline Tensor dispatch_round(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.round();
}
inline Tensor & dispatch_round_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.round_();
}
inline Tensor dispatch_rsqrt(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.rsqrt();
}
inline Tensor & dispatch_rsqrt_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.rsqrt_();
}
inline Tensor & dispatch_scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.scatter_(dim, index, value);
}
inline Tensor & dispatch_scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.scatter_(dim, index, src);
}
inline Tensor & dispatch_scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.scatter_add_(dim, index, src);
}
inline Tensor dispatch_select(Tensor & self, int64_t dim, int64_t index) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.select(dim, index);
}
inline Tensor & dispatch_set_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.set_();
}
inline Tensor & dispatch_set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.set_(sourceStorage, storage_offset, size, stride);
}
inline Tensor & dispatch_set_(Tensor & self, Storage & storage) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.set_(storage);
}
inline Tensor & dispatch_set_(Tensor & self, const Tensor & source) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.set_(source);
}
inline Tensor dispatch_sigmoid(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sigmoid();
}
inline Tensor & dispatch_sigmoid_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sigmoid_();
}
inline Tensor dispatch_sign(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sign();
}
inline Tensor & dispatch_sign_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sign_();
}
inline Tensor dispatch_sin(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sin();
}
inline Tensor & dispatch_sin_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sin_();
}
inline Tensor dispatch_sinh(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sinh();
}
inline Tensor & dispatch_sinh_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sinh_();
}
inline Tensor dispatch_slice(Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.slice(dim, start, end, step);
}
inline Tensor dispatch_smm(Tensor & self, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.smm(mat2);
}
inline std::tuple<Tensor,Tensor> dispatch_sort(Tensor & self, int64_t dim, bool descending) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sort(dim, descending);
}
inline std::vector<Tensor> dispatch_split(Tensor & self, int64_t split_size, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.split(split_size, dim);
}
inline Tensor dispatch_sqrt(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sqrt();
}
inline Tensor & dispatch_sqrt_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sqrt_();
}
inline Tensor dispatch_squeeze(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.squeeze();
}
inline Tensor dispatch_squeeze(Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.squeeze(dim);
}
inline Tensor & dispatch_squeeze_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.squeeze_();
}
inline Tensor & dispatch_squeeze_(Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.squeeze_(dim);
}
inline Tensor dispatch_sspaddmm(Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sspaddmm(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_sspaddmm(Scalar beta, Tensor & self, const Tensor & mat1, const Tensor & mat2) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sspaddmm(mat1, mat2, beta, 1);
}
inline Tensor dispatch_sspaddmm(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sspaddmm(mat1, mat2, beta, alpha);
}
inline Tensor dispatch_std(Tensor & self, bool unbiased) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.std(unbiased);
}
inline Tensor dispatch_std(Tensor & self, int64_t dim, bool unbiased, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.std(dim, unbiased, keepdim);
}
inline Tensor dispatch_stft(Tensor & self, int64_t frame_length, int64_t hop, int64_t fft_size, bool return_onesided, const Tensor & window, int64_t pad_end) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.stft(frame_length, hop, fft_size, return_onesided, window, pad_end);
}
inline int64_t dispatch_storage_offset(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.storage_offset();
}
inline Tensor dispatch_sub(Tensor & self, Scalar alpha, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sub(other, alpha);
}
inline Tensor dispatch_sub(Tensor & self, Scalar other, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sub(other, alpha);
}
inline Tensor dispatch_sub(Tensor & self, const Tensor & other, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sub(other, alpha);
}
inline Tensor & dispatch_sub_(Tensor & self, Scalar alpha, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sub_(other, alpha);
}
inline Tensor & dispatch_sub_(Tensor & self, Scalar other, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sub_(other, alpha);
}
inline Tensor & dispatch_sub_(Tensor & self, const Tensor & other, Scalar alpha) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sub_(other, alpha);
}
inline Tensor dispatch_sum(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sum();
}
inline Tensor dispatch_sum(Tensor & self, int64_t dim, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.sum(dim, keepdim);
}
inline std::tuple<Tensor,Tensor,Tensor> dispatch_svd(Tensor & self, bool some) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.svd(some);
}
inline std::tuple<Tensor,Tensor> dispatch_symeig(Tensor & self, bool eigenvectors, bool upper) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.symeig(eigenvectors, upper);
}
inline Tensor dispatch_t(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.t();
}
inline Tensor & dispatch_t_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.t_();
}
inline Tensor dispatch_take(Tensor & self, const Tensor & index) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.take(index);
}
inline Tensor dispatch_tan(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.tan();
}
inline Tensor & dispatch_tan_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.tan_();
}
inline Tensor dispatch_tanh(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.tanh();
}
inline Tensor & dispatch_tanh_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.tanh_();
}
inline Tensor dispatch_to_dense(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.to_dense();
}
inline std::tuple<Tensor,Tensor> dispatch_topk(Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.topk(k, dim, largest, sorted);
}
inline Tensor dispatch_trace(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.trace();
}
inline Tensor dispatch_transpose(Tensor & self, int64_t dim0, int64_t dim1) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.transpose(dim0, dim1);
}
inline Tensor & dispatch_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.transpose_(dim0, dim1);
}
inline Tensor dispatch_tril(Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.tril(diagonal);
}
inline Tensor & dispatch_tril_(Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.tril_(diagonal);
}
inline Tensor dispatch_triu(Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.triu(diagonal);
}
inline Tensor & dispatch_triu_(Tensor & self, int64_t diagonal) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.triu_(diagonal);
}
inline std::tuple<Tensor,Tensor> dispatch_trtrs(Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.trtrs(A, upper, transpose, unitriangular);
}
inline Tensor dispatch_trunc(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.trunc();
}
inline Tensor & dispatch_trunc_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.trunc_();
}
inline Tensor dispatch_type_as(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.type_as(other);
}
inline Tensor dispatch_unfold(Tensor & self, int64_t dimension, int64_t size, int64_t step) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.unfold(dimension, size, step);
}
inline Tensor & dispatch_uniform_(Tensor & self, double from, double to, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.uniform_(from, to, generator);
}
inline Tensor dispatch_unsqueeze(Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.unsqueeze(dim);
}
inline Tensor & dispatch_unsqueeze_(Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.unsqueeze_(dim);
}
inline Tensor dispatch_var(Tensor & self, bool unbiased) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.var(unbiased);
}
inline Tensor dispatch_var(Tensor & self, int64_t dim, bool unbiased, bool keepdim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.var(dim, unbiased, keepdim);
}
inline Tensor dispatch_view(Tensor & self, IntList size) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.view(size);
}
inline Tensor dispatch_view_as(Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.view_as(other);
}
inline Tensor dispatch_where(const Tensor & condition, Tensor & self, const Tensor & other) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(condition);
  return self.where(condition, other);
}
inline Tensor & dispatch_zero_(Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.zero_();
}

}} // namespace torch::autograd
