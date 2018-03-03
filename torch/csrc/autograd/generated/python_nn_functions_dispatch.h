#pragma once

// generated from tools/autograd/templates/python_nn_functions_dispatch.h

#include <ATen/ATen.h>
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/auto_gpu.h"

// Contains inline wrappers around ATen functions that release the GIL and
// switch to the correct CUDA device.

namespace torch { namespace autograd {

using namespace at;
using at::Generator;

inline Tensor & dispatch__sigmoid(const Tensor & self, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::_sigmoid_out(output, self);
}
inline Tensor dispatch__sigmoid(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::_sigmoid(self);
}
inline Tensor & dispatch__tanh(const Tensor & self, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::_tanh_out(output, self);
}
inline Tensor dispatch__tanh(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::_tanh(self);
}
inline Tensor & dispatch_adaptive_avg_pool2d(const Tensor & self, IntList output_size, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::adaptive_avg_pool2d_out(output, self, output_size);
}
inline Tensor dispatch_adaptive_avg_pool2d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::adaptive_avg_pool2d(self, output_size);
}
inline Tensor & dispatch_adaptive_avg_pool3d(const Tensor & self, IntList output_size, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::adaptive_avg_pool3d_out(output, self, output_size);
}
inline Tensor dispatch_adaptive_avg_pool3d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::adaptive_avg_pool3d(self, output_size);
}
inline std::tuple<Tensor &,Tensor &> dispatch_adaptive_max_pool2d(const Tensor & self, IntList output_size, Tensor & output, Tensor & indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::adaptive_max_pool2d_out(output, indices, self, output_size);
}
inline std::tuple<Tensor,Tensor> dispatch_adaptive_max_pool2d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::adaptive_max_pool2d(self, output_size);
}
inline std::tuple<Tensor &,Tensor &> dispatch_adaptive_max_pool3d(const Tensor & self, IntList output_size, Tensor & output, Tensor & indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::adaptive_max_pool3d_out(output, indices, self, output_size);
}
inline std::tuple<Tensor,Tensor> dispatch_adaptive_max_pool3d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::adaptive_max_pool3d(self, output_size);
}
inline Tensor & dispatch_avg_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::avg_pool2d_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
inline Tensor dispatch_avg_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
inline Tensor & dispatch_avg_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::avg_pool3d_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
inline Tensor dispatch_avg_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
inline Tensor & dispatch_binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::binary_cross_entropy_out(output, self, target, weight, size_average, reduce);
}
inline Tensor dispatch_binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::binary_cross_entropy(self, target, weight, size_average, reduce);
}
inline Tensor & dispatch_elu(const Tensor & self, Scalar alpha, Scalar scale, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::elu_out(output, self, alpha, scale);
}
inline Tensor dispatch_elu(const Tensor & self, Scalar alpha, Scalar scale) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::elu(self, alpha, scale);
}
inline Tensor & dispatch_elu_(Tensor self, Scalar alpha, Scalar scale) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::elu_(self, alpha, scale);
}
inline Tensor & dispatch_elu_forward_(Tensor self, Scalar alpha, Scalar scale) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::elu_forward_(self, alpha, scale);
}
inline std::tuple<Tensor &,Tensor &> dispatch_fractional_max_pool2d(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples, Tensor & output, Tensor & indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::fractional_max_pool2d_out(output, indices, self, kernel_size, output_size, random_samples);
}
inline std::tuple<Tensor,Tensor> dispatch_fractional_max_pool2d(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::fractional_max_pool2d(self, kernel_size, output_size, random_samples);
}
inline Tensor & dispatch_glu(const Tensor & self, int64_t dim, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::glu_out(output, self, dim);
}
inline Tensor dispatch_glu(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::glu(self, dim);
}
inline Tensor & dispatch_hardshrink(const Tensor & self, Scalar lambd, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::hardshrink_out(output, self, lambd);
}
inline Tensor dispatch_hardshrink(const Tensor & self, Scalar lambd) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::hardshrink(self, lambd);
}
inline Tensor & dispatch_hardtanh(const Tensor & self, Scalar min_val, Scalar max_val, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::hardtanh_out(output, self, min_val, max_val);
}
inline Tensor dispatch_hardtanh(const Tensor & self, Scalar min_val, Scalar max_val) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::hardtanh(self, min_val, max_val);
}
inline Tensor & dispatch_hardtanh_(Tensor self, Scalar min_val, Scalar max_val) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::hardtanh_(self, min_val, max_val);
}
inline Tensor & dispatch_hardtanh_forward_(Tensor self, Scalar min_val, Scalar max_val) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::hardtanh_forward_(self, min_val, max_val);
}
inline Tensor & dispatch_kl_div(const Tensor & self, const Tensor & target, bool size_average, bool reduce, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::kl_div_out(output, self, target, size_average, reduce);
}
inline Tensor dispatch_kl_div(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::kl_div(self, target, size_average, reduce);
}
inline Tensor & dispatch_l1_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::l1_loss_out(output, self, target, size_average, reduce);
}
inline Tensor dispatch_l1_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::l1_loss(self, target, size_average, reduce);
}
inline Tensor & dispatch_leaky_relu(const Tensor & self, Scalar negative_slope, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::leaky_relu_out(output, self, negative_slope);
}
inline Tensor dispatch_leaky_relu(const Tensor & self, Scalar negative_slope) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::leaky_relu(self, negative_slope);
}
inline Tensor & dispatch_leaky_relu_(Tensor self, Scalar negative_slope) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::leaky_relu_(self, negative_slope);
}
inline Tensor & dispatch_leaky_relu_forward_(Tensor self, Scalar negative_slope) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::leaky_relu_forward_(self, negative_slope);
}
inline Tensor & dispatch_log_sigmoid(const Tensor & self, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::log_sigmoid_out(output, self);
}
inline Tensor dispatch_log_sigmoid(const Tensor & self) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::log_sigmoid(self);
}
inline Tensor & dispatch_log_softmax(const Tensor & self, int64_t dim, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::log_softmax_out(output, self, dim);
}
inline Tensor dispatch_log_softmax(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::log_softmax(self, dim);
}
inline std::tuple<Tensor &,Tensor &> dispatch_max_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, Tensor & output, Tensor & indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::max_pool2d_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
inline std::tuple<Tensor,Tensor> dispatch_max_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
}
inline std::tuple<Tensor &,Tensor &> dispatch_max_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, Tensor & output, Tensor & indices) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::max_pool3d_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
inline std::tuple<Tensor,Tensor> dispatch_max_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
}
inline Tensor & dispatch_max_unpool2d(const Tensor & self, const Tensor & indices, IntList output_size, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::max_unpool2d_out(output, self, indices, output_size);
}
inline Tensor dispatch_max_unpool2d(const Tensor & self, const Tensor & indices, IntList output_size) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::max_unpool2d(self, indices, output_size);
}
inline Tensor & dispatch_max_unpool3d(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::max_unpool3d_out(output, self, indices, output_size, stride, padding);
}
inline Tensor dispatch_max_unpool3d(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::max_unpool3d(self, indices, output_size, stride, padding);
}
inline Tensor & dispatch_mse_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::mse_loss_out(output, self, target, size_average, reduce);
}
inline Tensor dispatch_mse_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::mse_loss(self, target, size_average, reduce);
}
inline Tensor & dispatch_multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::multi_margin_loss_out(output, self, target, p, margin, weight, size_average);
}
inline Tensor dispatch_multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::multi_margin_loss(self, target, p, margin, weight, size_average);
}
inline Tensor & dispatch_multilabel_margin_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::multilabel_margin_loss_out(output, self, target, size_average, reduce);
}
inline Tensor dispatch_multilabel_margin_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::multilabel_margin_loss(self, target, size_average, reduce);
}
inline Tensor & dispatch_nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::nll_loss_out(output, self, target, weight, size_average, ignore_index, reduce);
}
inline Tensor dispatch_nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::nll_loss(self, target, weight, size_average, ignore_index, reduce);
}
inline Tensor & dispatch_nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::nll_loss2d_out(output, self, target, weight, size_average, ignore_index, reduce);
}
inline Tensor dispatch_nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::nll_loss2d(self, target, weight, size_average, ignore_index, reduce);
}
inline Tensor & dispatch_prelu(const Tensor & self, const Tensor & weight, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::prelu_out(output, self, weight);
}
inline Tensor dispatch_prelu(const Tensor & self, const Tensor & weight) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::prelu(self, weight);
}
inline Tensor & dispatch_reflection_pad1d(const Tensor & self, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::reflection_pad1d_out(output, self, padding);
}
inline Tensor dispatch_reflection_pad1d(const Tensor & self, IntList padding) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::reflection_pad1d(self, padding);
}
inline Tensor & dispatch_reflection_pad2d(const Tensor & self, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::reflection_pad2d_out(output, self, padding);
}
inline Tensor dispatch_reflection_pad2d(const Tensor & self, IntList padding) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::reflection_pad2d(self, padding);
}
inline Tensor & dispatch_replication_pad1d(const Tensor & self, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::replication_pad1d_out(output, self, padding);
}
inline Tensor dispatch_replication_pad1d(const Tensor & self, IntList padding) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::replication_pad1d(self, padding);
}
inline Tensor & dispatch_replication_pad2d(const Tensor & self, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::replication_pad2d_out(output, self, padding);
}
inline Tensor dispatch_replication_pad2d(const Tensor & self, IntList padding) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::replication_pad2d(self, padding);
}
inline Tensor & dispatch_replication_pad3d(const Tensor & self, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::replication_pad3d_out(output, self, padding);
}
inline Tensor dispatch_replication_pad3d(const Tensor & self, IntList padding) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::replication_pad3d(self, padding);
}
inline Tensor & dispatch_rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::rrelu_with_noise_out(output, self, noise, lower, upper, training, generator);
}
inline Tensor dispatch_rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::rrelu_with_noise(self, noise, lower, upper, training, generator);
}
inline Tensor & dispatch_rrelu_with_noise_(Tensor self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::rrelu_with_noise_(self, noise, lower, upper, training, generator);
}
inline Tensor & dispatch_rrelu_with_noise_forward_(Tensor self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::rrelu_with_noise_forward_(self, noise, lower, upper, training, generator);
}
inline Tensor & dispatch_smooth_l1_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::smooth_l1_loss_out(output, self, target, size_average, reduce);
}
inline Tensor dispatch_smooth_l1_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::smooth_l1_loss(self, target, size_average, reduce);
}
inline Tensor & dispatch_soft_margin_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::soft_margin_loss_out(output, self, target, size_average, reduce);
}
inline Tensor dispatch_soft_margin_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::soft_margin_loss(self, target, size_average, reduce);
}
inline Tensor & dispatch_softmax(const Tensor & self, int64_t dim, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::softmax_out(output, self, dim);
}
inline Tensor dispatch_softmax(const Tensor & self, int64_t dim) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::softmax(self, dim);
}
inline Tensor & dispatch_softplus(const Tensor & self, Scalar beta, Scalar threshold, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::softplus_out(output, self, beta, threshold);
}
inline Tensor dispatch_softplus(const Tensor & self, Scalar beta, Scalar threshold) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::softplus(self, beta, threshold);
}
inline Tensor & dispatch_softshrink(const Tensor & self, Scalar lambd, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::softshrink_out(output, self, lambd);
}
inline Tensor dispatch_softshrink(const Tensor & self, Scalar lambd) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::softshrink(self, lambd);
}
inline Tensor & dispatch_thnn_batch_norm(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::thnn_batch_norm_out(output, self, weight, bias, running_mean, running_var, training, momentum, eps);
}
inline Tensor dispatch_thnn_batch_norm(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::thnn_batch_norm(self, weight, bias, running_mean, running_var, training, momentum, eps);
}
inline Tensor & dispatch_thnn_conv2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::thnn_conv2d_out(output, self, weight, kernel_size, bias, stride, padding);
}
inline Tensor dispatch_thnn_conv2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::thnn_conv2d(self, weight, kernel_size, bias, stride, padding);
}
inline Tensor & dispatch_thnn_conv3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::thnn_conv3d_out(output, self, weight, kernel_size, bias, stride, padding);
}
inline Tensor dispatch_thnn_conv3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::thnn_conv3d(self, weight, kernel_size, bias, stride, padding);
}
inline Tensor & dispatch_thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::thnn_conv_depthwise2d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}
inline Tensor dispatch_thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::thnn_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
}
inline Tensor & dispatch_thnn_conv_dilated2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::thnn_conv_dilated2d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}
inline Tensor dispatch_thnn_conv_dilated2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::thnn_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
}
inline Tensor & dispatch_thnn_conv_dilated3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::thnn_conv_dilated3d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}
inline Tensor dispatch_thnn_conv_dilated3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::thnn_conv_dilated3d(self, weight, kernel_size, bias, stride, padding, dilation);
}
inline Tensor & dispatch_thnn_conv_transpose2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::thnn_conv_transpose2d_out(output, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
inline Tensor dispatch_thnn_conv_transpose2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::thnn_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
inline Tensor & dispatch_thnn_conv_transpose3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::thnn_conv_transpose3d_out(output, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
inline Tensor dispatch_thnn_conv_transpose3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::thnn_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
inline Tensor & dispatch_threshold(const Tensor & self, Scalar threshold, Scalar value, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::threshold_out(output, self, threshold, value);
}
inline Tensor dispatch_threshold(const Tensor & self, Scalar threshold, Scalar value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::threshold(self, threshold, value);
}
inline Tensor & dispatch_threshold_(Tensor self, Scalar threshold, Scalar value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::threshold_(self, threshold, value);
}
inline Tensor & dispatch_threshold_forward_(Tensor self, Scalar threshold, Scalar value) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::threshold_forward_(self, threshold, value);
}
inline Tensor & dispatch_upsample_bilinear2d(const Tensor & self, IntList output_size, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::upsample_bilinear2d_out(output, self, output_size);
}
inline Tensor dispatch_upsample_bilinear2d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::upsample_bilinear2d(self, output_size);
}
inline Tensor & dispatch_upsample_linear1d(const Tensor & self, IntList output_size, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::upsample_linear1d_out(output, self, output_size);
}
inline Tensor dispatch_upsample_linear1d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::upsample_linear1d(self, output_size);
}
inline Tensor & dispatch_upsample_nearest1d(const Tensor & self, int64_t scale_factor, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::upsample_nearest1d_out(output, self, scale_factor);
}
inline Tensor dispatch_upsample_nearest1d(const Tensor & self, int64_t scale_factor) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::upsample_nearest1d(self, scale_factor);
}
inline Tensor & dispatch_upsample_nearest2d(const Tensor & self, int64_t scale_factor, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::upsample_nearest2d_out(output, self, scale_factor);
}
inline Tensor dispatch_upsample_nearest2d(const Tensor & self, int64_t scale_factor) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::upsample_nearest2d(self, scale_factor);
}
inline Tensor & dispatch_upsample_nearest3d(const Tensor & self, int64_t scale_factor, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::upsample_nearest3d_out(output, self, scale_factor);
}
inline Tensor dispatch_upsample_nearest3d(const Tensor & self, int64_t scale_factor) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::upsample_nearest3d(self, scale_factor);
}
inline Tensor & dispatch_upsample_trilinear3d(const Tensor & self, IntList output_size, Tensor output) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(output);
  return at::upsample_trilinear3d_out(output, self, output_size);
}
inline Tensor dispatch_upsample_trilinear3d(const Tensor & self, IntList output_size) {

  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return at::upsample_trilinear3d(self, output_size);
}

}} // namespace torch::autograd
