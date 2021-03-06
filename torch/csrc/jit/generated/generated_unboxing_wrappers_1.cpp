#include "torch/csrc/jit/runtime/operator.h"
#include "torch/csrc/jit/runtime/custom_operator.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"

#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// @generated from tools\jit\templates/generated_unboxing_wrappers.cpp

// This file contains manual unboxing wrappers for ops that aren't
// use_c10_dispatcher: full because the templated unboxing logic in c10 doesn't
// support them yet. The ultimate goal is to make all ops use the templated
// unboxing and delete this codegen file.

// NOTE [Sharded File]: This file is generated in a sharded fashion to speed up
// incremental rebuilds. See the comment at the top of
// templates/VariableType.cpp for an analogous, in-depth discussion.

namespace torch { namespace jit {

using autograd::Variable;
using autograd::variable_list;
using at::Scalar;
using at::ScalarType;
using at::Tensor;
using at::TensorOptions;
using at::DeviceGuard;
using at::MemoryFormat;

using ::c10::fmap;
using ::c10::filter;
using c10::OperatorKernel;
using c10::OperatorHandle;
using c10::KernelFunction;
using c10::RegistrationHandleRAII;
using c10::Stack;

namespace {

template<class Return, class... Args>
Return callUnboxedKernel(OperatorKernel* unboxedKernel, Args... args) {
  using FuncType = Return (Args...);
  auto* typedUnboxedKernel = static_cast<c10::impl::WrapFunctionIntoRuntimeFunctor<FuncType*>*>(unboxedKernel);
  return (*typedUnboxedKernel)(std::forward<Args>(args)...);
}

// TODO: remove the toOptionalTensor and toListOfOptionalTensor
// when we remove the undefined tensor semantic from TH

// XXX: This function is to specialize IValue for tensor type in
// interpreter, it should only be used in this file
at::Tensor toOptionalTensor(const IValue& v) {
  if (v.isNone()) {
    return at::Tensor();
  }
  return v.toTensor();
}

// XXX: This function is to specialize IValue for list of optional
// tensor type in interpreter, it should only be used in this file
std::vector<Tensor> toListOfOptionalTensor(const IValue& v) {
  // v is a list of optional tensor, loop over as generic list
  auto vlist = v.toListRef();
  std::vector<Tensor> res;

  for (const IValue &v: vlist) {
    res.emplace_back(toOptionalTensor(v));
  }
  return res;
}

template<size_t N>
std::array<bool, N> as_bool_array(const c10::List<bool>& list) {
  std::array<bool, N> res;
  AT_ASSERT(list.size() == N);
  std::copy(list.begin(), list.end(), res.begin());
  return res;
}

KernelFunction::InternalBoxedKernelFunction *DUMMY_OPERATION =
  [](c10::OperatorKernel *, const c10::OperatorHandle &, std::vector<c10::IValue> *) -> void {
    TORCH_CHECK(false, "Operator has been stripped in the custom build.")
  };

class Registerer final {
public:
  Registerer&& op(const std::string& schemaStr, KernelFunction::InternalBoxedKernelFunction* boxed_kernel_wrapper) && {
    static auto& dispatcher = c10::Dispatcher::singleton();
    auto schema = parseSchema(schemaStr);
    schema.setAliasAnalysis(AliasAnalysisKind::FROM_SCHEMA);
    c10::OperatorName name = schema.operator_name();
    RegistrationHandleRAII registration = dispatcher.registerName(name);
    auto op = dispatcher.findOp(name).value();
    registrationHandles_.push_back(std::move(registration));
    dispatcher.setManuallyBoxedKernelFor_(op, boxed_kernel_wrapper);
    return std::move(*this);
  }

  Registerer() = default;
  Registerer(const Registerer&) = delete;
  Registerer& operator=(const Registerer&) = delete;
  Registerer(Registerer&&) noexcept = default;
  Registerer& operator=(Registerer&&) noexcept = default;
private:
  std::vector<RegistrationHandleRAII> registrationHandles_;
};

static auto registry = Registerer()
  // Generated operators
    .op("aten::__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::__irshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::__irshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::_addmv_impl_(Tensor(a!) self, Tensor self2, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          (std::move(peek(*stack, 3, 6))).toTensor(),
          (std::move(peek(*stack, 4, 6))).toScalar(),
          (std::move(peek(*stack, 5, 6))).toScalar());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::_addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toTensor(),
          (std::move(peek(*stack, 3, 5))).toScalar(),
          (std::move(peek(*stack, 4, 5))).toScalar());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool>(unboxedKernel, (std::move(peek(*stack, 0, 9))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 1, 9)))),
          toOptionalTensor((std::move(peek(*stack, 2, 9)))),
          toOptionalTensor((std::move(peek(*stack, 3, 9)))),
          toOptionalTensor((std::move(peek(*stack, 4, 9)))),
          (std::move(peek(*stack, 5, 9))).toBool(),
          (std::move(peek(*stack, 6, 9))).toDouble(),
          (std::move(peek(*stack, 7, 9))).toDouble(),
          (std::move(peek(*stack, 8, 9))).toBool());
          drop(*stack, 9);
          pack(*stack, std::move(result_));
      })
    .op("aten::_bmm.out(Tensor self, Tensor mat2, *, bool deterministic=False, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toBool());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>, const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 15))).toTensor(),
          (std::move(peek(*stack, 1, 15))).toTensorVector(),
          (std::move(peek(*stack, 2, 15))).toInt(),
          toOptionalTensor((std::move(peek(*stack, 3, 15)))),
          (std::move(peek(*stack, 4, 15))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 5, 15)))),
          (std::move(peek(*stack, 6, 15))).toInt(),
          (std::move(peek(*stack, 7, 15))).toInt(),
          (std::move(peek(*stack, 8, 15))).toInt(),
          (std::move(peek(*stack, 9, 15))).toBool(),
          (std::move(peek(*stack, 10, 15))).toDouble(),
          (std::move(peek(*stack, 11, 15))).toBool(),
          (std::move(peek(*stack, 12, 15))).toBool(),
          (std::move(peek(*stack, 13, 15))).toIntVector(),
          toOptionalTensor((std::move(peek(*stack, 14, 15)))));
          drop(*stack, 15);
          pack(*stack, std::move(result_));
      })
    .op("aten::_embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, bool, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 11))).toTensor(),
          (std::move(peek(*stack, 1, 11))).toTensor(),
          (std::move(peek(*stack, 2, 11))).toTensor(),
          (std::move(peek(*stack, 3, 11))).toTensor(),
          (std::move(peek(*stack, 4, 11))).toTensor(),
          (std::move(peek(*stack, 5, 11))).toTensor(),
          (std::move(peek(*stack, 6, 11))).toInt(),
          (std::move(peek(*stack, 7, 11))).toBool(),
          (std::move(peek(*stack, 8, 11))).toInt(),
          (std::move(peek(*stack, 9, 11))).toBool(),
          toOptionalTensor((std::move(peek(*stack, 10, 11)))));
          drop(*stack, 11);
          pack(*stack, std::move(result_));
      })
    .op("aten::_empty_per_channel_affine_quantized(int[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 4, 9))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 5, 9))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 6, 9))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 7, 9))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::_empty_per_channel_affine_quantized((std::move(peek(*stack, 0, 9))).toIntVector(),
          (std::move(peek(*stack, 1, 9))).toTensor(),
          (std::move(peek(*stack, 2, 9))).toTensor(),
          (std::move(peek(*stack, 3, 9))).toInt(),
          options,
          (std::move(peek(*stack, 8, 9))).toOptional<c10::MemoryFormat>());
          #else
              auto result_ = torch::_empty_per_channel_affine_quantized((std::move(peek(*stack, 0, 9))).toIntVector(),
          (std::move(peek(*stack, 1, 9))).toTensor(),
          (std::move(peek(*stack, 2, 9))).toTensor(),
          (std::move(peek(*stack, 3, 9))).toInt(),
          options,
          (std::move(peek(*stack, 8, 9))).toOptional<c10::MemoryFormat>());
          #endif
          drop(*stack, 9);
          pack(*stack, std::move(result_));
      })
    .op("aten::_index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 4))).toInt(),
          (std::move(peek(*stack, 2, 4))).toTensor(),
          (std::move(peek(*stack, 3, 4))).toTensor());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::_logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sobol_engine_draw(Tensor quasi, int n, Tensor sobolstate, int dimension, int num_generated, ScalarType? dtype) -> (Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor>, const Tensor &, int64_t, const Tensor &, int64_t, int64_t, c10::optional<ScalarType>>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toInt(),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          (std::move(peek(*stack, 3, 6))).toInt(),
          (std::move(peek(*stack, 4, 6))).toInt(),
          (std::move(peek(*stack, 5, 6))).toOptional<ScalarType>());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::_thnn_fused_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool>(unboxedKernel, toOptionalTensor((std::move(peek(*stack, 0, 6)))),
          toOptionalTensor((std::move(peek(*stack, 1, 6)))),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          (std::move(peek(*stack, 3, 6))).toTensor(),
          (std::move(peek(*stack, 4, 6))).toTensor(),
          (std::move(peek(*stack, 5, 6))).toBool());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::absolute_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::acos_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::adaptive_avg_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toIntVector());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::adaptive_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toTensor());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toScalar());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toTensor(),
          (std::move(peek(*stack, 3, 4))).toScalar());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toTensor(),
          (std::move(peek(*stack, 3, 5))).toScalar(),
          (std::move(peek(*stack, 4, 5))).toScalar());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toTensor(),
          (std::move(peek(*stack, 3, 5))).toScalar(),
          (std::move(peek(*stack, 4, 5))).toScalar());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          (std::move(peek(*stack, 3, 6))).toScalar(),
          (std::move(peek(*stack, 4, 6))).toScalar());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, double, bool>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toDouble(),
          (std::move(peek(*stack, 2, 3))).toBool());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 5))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::arange((std::move(peek(*stack, 0, 5))).toScalar(),
          options);
          #else
              auto result_ = torch::arange((std::move(peek(*stack, 0, 5))).toScalar(),
          options);
          #endif
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::arange((std::move(peek(*stack, 0, 6))).toScalar(),
          (std::move(peek(*stack, 1, 6))).toScalar(),
          options);
          #else
              auto result_ = torch::arange((std::move(peek(*stack, 0, 6))).toScalar(),
          (std::move(peek(*stack, 1, 6))).toScalar(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 4, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 5, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::arange((std::move(peek(*stack, 0, 7))).toScalar(),
          (std::move(peek(*stack, 1, 7))).toScalar(),
          (std::move(peek(*stack, 2, 7))).toScalar(),
          options);
          #else
              auto result_ = torch::arange((std::move(peek(*stack, 0, 7))).toScalar(),
          (std::move(peek(*stack, 1, 7))).toScalar(),
          (std::move(peek(*stack, 2, 7))).toScalar(),
          options);
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::as_strided_(Tensor(a!) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 4))).toIntVector(),
          (std::move(peek(*stack, 2, 4))).toIntVector(),
          (std::move(peek(*stack, 3, 4))).toOptional<int64_t>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::atanh_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 7, 8))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toIntVector(),
          (std::move(peek(*stack, 2, 8))).toIntVector(),
          (std::move(peek(*stack, 3, 8))).toIntVector(),
          (std::move(peek(*stack, 4, 8))).toBool(),
          (std::move(peek(*stack, 5, 8))).toBool(),
          (std::move(peek(*stack, 6, 8))).toOptional<int64_t>());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 8, 9))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 9))).toTensor(),
          (std::move(peek(*stack, 1, 9))).toTensor(),
          (std::move(peek(*stack, 2, 9))).toIntVector(),
          (std::move(peek(*stack, 3, 9))).toIntVector(),
          (std::move(peek(*stack, 4, 9))).toIntVector(),
          (std::move(peek(*stack, 5, 9))).toBool(),
          (std::move(peek(*stack, 6, 9))).toBool(),
          (std::move(peek(*stack, 7, 9))).toOptional<int64_t>());
          drop(*stack, 9);
          pack(*stack, std::move(result_));
      })
    .op("aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          (std::move(peek(*stack, 3, 6))).toScalar(),
          (std::move(peek(*stack, 4, 6))).toScalar());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::batch_norm_elemt.out(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 6, 7))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 7))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 1, 7)))),
          toOptionalTensor((std::move(peek(*stack, 2, 7)))),
          (std::move(peek(*stack, 3, 7))).toTensor(),
          (std::move(peek(*stack, 4, 7))).toTensor(),
          (std::move(peek(*stack, 5, 7))).toDouble());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          (std::move(peek(*stack, 2, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 8)))),
          toOptionalTensor((std::move(peek(*stack, 4, 8)))),
          (std::move(peek(*stack, 5, 8))).toDouble(),
          (std::move(peek(*stack, 6, 8))).toDouble(),
          (std::move(peek(*stack, 7, 8))).toInt());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, Tensor counts) -> (Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          (std::move(peek(*stack, 2, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 8)))),
          toOptionalTensor((std::move(peek(*stack, 4, 8)))),
          (std::move(peek(*stack, 5, 8))).toDouble(),
          (std::move(peek(*stack, 6, 8))).toDouble(),
          (std::move(peek(*stack, 7, 8))).toTensor());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, c10::optional<Generator>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 4)))));
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 5)))),
          (std::move(peek(*stack, 4, 5))).toInt());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 5)))),
          toOptionalTensor((std::move(peek(*stack, 3, 5)))),
          (std::move(peek(*stack, 4, 5))).toInt());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_and.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_or_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_or_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_xor_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_xor_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::bucketize.Tensor_out(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, bool, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toBool(),
          (std::move(peek(*stack, 3, 5))).toBool());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, TensorList, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensorVector(),
          (std::move(peek(*stack, 1, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::ceil_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::celu_(Tensor(a!) self, Scalar alpha=1.0) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::cholesky_inverse.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toBool());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toOptional<Scalar>(),
          (std::move(peek(*stack, 2, 4))).toOptional<Scalar>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, c10::optional<MemoryFormat>>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          (std::move(peek(*stack, 1, 2))).toOptional<c10::MemoryFormat>());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::col2im.out(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 6, 7))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toIntVector(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          (std::move(peek(*stack, 3, 7))).toIntVector(),
          (std::move(peek(*stack, 4, 7))).toIntVector(),
          (std::move(peek(*stack, 5, 7))).toIntVector());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::conj.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 7)))),
          (std::move(peek(*stack, 3, 7))).toIntVector(),
          (std::move(peek(*stack, 4, 7))).toIntVector(),
          (std::move(peek(*stack, 5, 7))).toIntVector(),
          (std::move(peek(*stack, 6, 7))).toInt());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 8)))),
          (std::move(peek(*stack, 3, 8))).toIntVector(),
          (std::move(peek(*stack, 4, 8))).toIntVector(),
          (std::move(peek(*stack, 5, 8))).toIntVector(),
          (std::move(peek(*stack, 6, 8))).toInt(),
          (std::move(peek(*stack, 7, 8))).toIntVector());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 9))).toTensor(),
          (std::move(peek(*stack, 1, 9))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 9)))),
          (std::move(peek(*stack, 3, 9))).toIntVector(),
          (std::move(peek(*stack, 4, 9))).toIntVector(),
          (std::move(peek(*stack, 5, 9))).toIntVector(),
          (std::move(peek(*stack, 6, 9))).toBool(),
          (std::move(peek(*stack, 7, 9))).toIntVector(),
          (std::move(peek(*stack, 8, 9))).toInt());
          drop(*stack, 9);
          pack(*stack, std::move(result_));
      })
    .op("aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, bool>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toBool());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 8)))),
          toOptionalTensor((std::move(peek(*stack, 3, 8)))),
          toOptionalTensor((std::move(peek(*stack, 4, 8)))),
          (std::move(peek(*stack, 5, 8))).toBool(),
          (std::move(peek(*stack, 6, 8))).toDouble(),
          (std::move(peek(*stack, 7, 8))).toDouble());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t, c10::optional<ScalarType>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toInt(),
          (std::move(peek(*stack, 2, 4))).toOptional<ScalarType>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t, c10::optional<ScalarType>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toInt(),
          (std::move(peek(*stack, 2, 4))).toOptional<ScalarType>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::deg2rad_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::detach_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::diag.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar, Scalar, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 4))).toScalar(),
          (std::move(peek(*stack, 2, 4))).toScalar(),
          (std::move(peek(*stack, 3, 4))).toScalar());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::empty((std::move(peek(*stack, 0, 6))).toIntVector(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #else
              auto result_ = torch::empty((std::move(peek(*stack, 0, 6))).toIntVector(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::empty_like((std::move(peek(*stack, 0, 6))).toTensor(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #else
              auto result_ = torch::empty_like((std::move(peek(*stack, 0, 6))).toTensor(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::empty_strided((std::move(peek(*stack, 0, 6))).toIntVector(),
          (std::move(peek(*stack, 1, 6))).toIntVector(),
          options);
          #else
              auto result_ = torch::empty_strided((std::move(peek(*stack, 0, 6))).toIntVector(),
          (std::move(peek(*stack, 1, 6))).toIntVector(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::erf_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::erfc_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::erfinv_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::expm1_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, double, bool>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toDouble(),
          (std::move(peek(*stack, 2, 3))).toBool());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar, bool>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toScalar(),
          (std::move(peek(*stack, 2, 3))).toBool());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::fractional_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toIntVector(),
          (std::move(peek(*stack, 3, 6))).toIntVector(),
          (std::move(peek(*stack, 4, 6))).toTensor());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::frobenius_norm.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toIntVector(),
          (std::move(peek(*stack, 2, 4))).toBool());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::full.out(int[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, IntArrayRef, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toIntVector(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, double, c10::optional<Generator>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toDouble(),
          (std::move(peek(*stack, 2, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::ger.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::glu_backward.grad_input(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toInt());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &, double, bool>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toInt(),
          toOptionalTensor((std::move(peek(*stack, 2, 6)))),
          toOptionalTensor((std::move(peek(*stack, 3, 6)))),
          (std::move(peek(*stack, 4, 6))).toDouble(),
          (std::move(peek(*stack, 5, 6))).toBool());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 5))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::hann_window((std::move(peek(*stack, 0, 5))).toInt(),
          options);
          #else
              auto result_ = torch::hann_window((std::move(peek(*stack, 0, 5))).toInt(),
          options);
          #endif
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::hann_window((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toBool(),
          options);
          #else
              auto result_ = torch::hann_window((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toBool(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::hardsigmoid_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toScalar(),
          (std::move(peek(*stack, 2, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::hspmm.out(Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::im2col.out(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toIntVector(),
          (std::move(peek(*stack, 2, 6))).toIntVector(),
          (std::move(peek(*stack, 3, 6))).toIntVector(),
          (std::move(peek(*stack, 4, 6))).toIntVector());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, TensorList, const Tensor &, bool>(unboxedKernel, (std::move(peek(*stack, 0, 4))).toTensor(),
          toListOfOptionalTensor((std::move(peek(*stack, 1, 4)))),
          (std::move(peek(*stack, 2, 4))).toTensor(),
          (std::move(peek(*stack, 3, 4))).toBool());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toTensor());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toScalar());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::lgamma_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::linspace(Scalar start, Scalar end, int steps=100, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 4, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 5, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::linspace((std::move(peek(*stack, 0, 7))).toScalar(),
          (std::move(peek(*stack, 1, 7))).toScalar(),
          (std::move(peek(*stack, 2, 7))).toInt(),
          options);
          #else
              auto result_ = torch::linspace((std::move(peek(*stack, 0, 7))).toScalar(),
          (std::move(peek(*stack, 1, 7))).toScalar(),
          (std::move(peek(*stack, 2, 7))).toInt(),
          options);
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::log10_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::log_sigmoid_backward.grad_input(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toTensor());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::logaddexp2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::logcumsumexp(Tensor self, int dim) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          (std::move(peek(*stack, 1, 2))).toInt());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::logical_and_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::logical_not_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::logspace.out(Scalar start, Scalar end, int steps=100, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar, Scalar, int64_t, double>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toScalar(),
          (std::move(peek(*stack, 1, 5))).toScalar(),
          (std::move(peek(*stack, 2, 5))).toInt(),
          (std::move(peek(*stack, 3, 5))).toDouble());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toIntVector(),
          (std::move(peek(*stack, 2, 4))).toBool());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::lu_solve.out(Tensor self, Tensor LU_data, Tensor LU_pivots, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toTensor());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::max_pool3d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 8, 9))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 9))).toTensor(),
          (std::move(peek(*stack, 1, 9))).toTensor(),
          (std::move(peek(*stack, 2, 9))).toIntVector(),
          (std::move(peek(*stack, 3, 9))).toIntVector(),
          (std::move(peek(*stack, 4, 9))).toIntVector(),
          (std::move(peek(*stack, 5, 9))).toIntVector(),
          (std::move(peek(*stack, 6, 9))).toBool(),
          (std::move(peek(*stack, 7, 9))).toTensor());
          drop(*stack, 9);
          pack(*stack, std::move(result_));
      })
    .op("aten::max_unpool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toTensor(),
          (std::move(peek(*stack, 3, 5))).toIntVector());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toIntVector(),
          (std::move(peek(*stack, 2, 5))).toBool(),
          (std::move(peek(*stack, 3, 5))).toOptional<ScalarType>());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::miopen_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          (std::move(peek(*stack, 2, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 8)))),
          toOptionalTensor((std::move(peek(*stack, 4, 8)))),
          toOptionalTensor((std::move(peek(*stack, 5, 8)))),
          toOptionalTensor((std::move(peek(*stack, 6, 8)))),
          (std::move(peek(*stack, 7, 8))).toDouble());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toTensor(),
          (std::move(peek(*stack, 3, 7))).toScalar(),
          (std::move(peek(*stack, 4, 7))).toScalar(),
          toOptionalTensor((std::move(peek(*stack, 5, 7)))),
          (std::move(peek(*stack, 6, 7))).toInt());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::multilabel_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          (std::move(peek(*stack, 3, 6))).toInt(),
          (std::move(peek(*stack, 4, 6))).toTensor());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, int64_t, bool, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toInt(),
          (std::move(peek(*stack, 2, 4))).toBool(),
          (std::move(peek(*stack, 3, 4))).toOptional<at::Generator>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 1, 8)))),
          toOptionalTensor((std::move(peek(*stack, 2, 8)))),
          toOptionalTensor((std::move(peek(*stack, 3, 8)))),
          toOptionalTensor((std::move(peek(*stack, 4, 8)))),
          (std::move(peek(*stack, 5, 8))).toBool(),
          (std::move(peek(*stack, 6, 8))).toDouble(),
          (std::move(peek(*stack, 7, 8))).toDouble());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int N, int C, int HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, int64_t, std::array<bool,3>>(unboxedKernel, (std::move(peek(*stack, 0, 10))).toTensor(),
          (std::move(peek(*stack, 1, 10))).toTensor(),
          (std::move(peek(*stack, 2, 10))).toTensor(),
          (std::move(peek(*stack, 3, 10))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 4, 10)))),
          (std::move(peek(*stack, 5, 10))).toInt(),
          (std::move(peek(*stack, 6, 10))).toInt(),
          (std::move(peek(*stack, 7, 10))).toInt(),
          (std::move(peek(*stack, 8, 10))).toInt(),
          as_bool_array<3>((std::move(peek(*stack, 9, 10))).toBoolList()));
          drop(*stack, 10);
          pack(*stack, std::move(result_));
      })
    .op("aten::native_layer_norm(Tensor input, Tensor? weight, Tensor? bias, int M, int N, float eps) -> (Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 1, 6)))),
          toOptionalTensor((std::move(peek(*stack, 2, 6)))),
          (std::move(peek(*stack, 3, 6))).toInt(),
          (std::move(peek(*stack, 4, 6))).toInt(),
          (std::move(peek(*stack, 5, 6))).toDouble());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::neg_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::new_zeros(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          auto result_ = ((std::move(peek(*stack, 0, 6))).toTensor()).new_zeros((std::move(peek(*stack, 1, 6))).toIntVector(),
          options);
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::nll_loss2d.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 6)))),
          (std::move(peek(*stack, 3, 6))).toInt(),
          (std::move(peek(*stack, 4, 6))).toInt());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::nll_loss.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 6)))),
          (std::move(peek(*stack, 3, 6))).toInt(),
          (std::move(peek(*stack, 4, 6))).toInt());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toOptional<Scalar>(),
          (std::move(peek(*stack, 2, 5))).toIntVector(),
          (std::move(peek(*stack, 3, 5))).toBool());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toOptional<Scalar>(),
          (std::move(peek(*stack, 2, 6))).toIntVector(),
          (std::move(peek(*stack, 3, 6))).toBool(),
          (std::move(peek(*stack, 4, 6))).toScalarType());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, double, const Tensor &, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toDouble(),
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, double, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toDouble(),
          (std::move(peek(*stack, 2, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::normal.float_float(float mean, float std, int[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 4, 8))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 5, 8))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 6, 8))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 7, 8))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::normal((std::move(peek(*stack, 0, 8))).toDouble(),
          (std::move(peek(*stack, 1, 8))).toDouble(),
          (std::move(peek(*stack, 2, 8))).toIntVector(),
          (std::move(peek(*stack, 3, 8))).toOptional<at::Generator>(),
          options);
          #else
              auto result_ = torch::normal((std::move(peek(*stack, 0, 8))).toDouble(),
          (std::move(peek(*stack, 1, 8))).toDouble(),
          (std::move(peek(*stack, 2, 8))).toIntVector(),
          (std::move(peek(*stack, 3, 8))).toOptional<at::Generator>(),
          options);
          #endif
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::orgqr.out(Tensor self, Tensor input2, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::ormqr.out(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          (std::move(peek(*stack, 3, 6))).toBool(),
          (std::move(peek(*stack, 4, 6))).toBool());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toScalar(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::quantize_per_tensor.tensors(Tensor[] tensors, Tensor scales, Tensor zero_points, ScalarType dtype) -> Tensor[]",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::vector<Tensor>, TensorList, const Tensor &, const Tensor &, ScalarType>(unboxedKernel, (std::move(peek(*stack, 0, 4))).toTensorVector(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toTensor(),
          (std::move(peek(*stack, 3, 4))).toScalarType());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, double, int64_t, ScalarType>(unboxedKernel, (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toDouble(),
          (std::move(peek(*stack, 2, 4))).toInt(),
          (std::move(peek(*stack, 3, 4))).toScalarType());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::rad2deg_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::rand(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 5))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::rand((std::move(peek(*stack, 0, 5))).toIntVector(),
          options);
          #else
              auto result_ = torch::rand((std::move(peek(*stack, 0, 5))).toIntVector(),
          options);
          #endif
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::rand.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::rand((std::move(peek(*stack, 0, 6))).toIntVector(),
          (std::move(peek(*stack, 1, 6))).toOptional<at::Generator>(),
          options);
          #else
              auto result_ = torch::rand((std::move(peek(*stack, 0, 6))).toIntVector(),
          (std::move(peek(*stack, 1, 6))).toOptional<at::Generator>(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::rand_like((std::move(peek(*stack, 0, 6))).toTensor(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #else
              auto result_ = torch::rand_like((std::move(peek(*stack, 0, 6))).toTensor(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::randn.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toIntVector());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::randn.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, IntArrayRef, c10::optional<Generator>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toIntVector(),
          (std::move(peek(*stack, 1, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, c10::optional<Generator>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toOptional<at::Generator>());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, c10::optional<Generator>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toInt(),
          (std::move(peek(*stack, 2, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, c10::optional<int64_t>, c10::optional<Generator>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 4))).toInt(),
          (std::move(peek(*stack, 2, 4))).toOptional<int64_t>(),
          (std::move(peek(*stack, 3, 4))).toOptional<at::Generator>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::range.out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar, Scalar, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toScalar(),
          (std::move(peek(*stack, 1, 4))).toScalar(),
          (std::move(peek(*stack, 2, 4))).toScalar());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::reflection_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toIntVector());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::reflection_pad2d_backward.grad_input(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toIntVector());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::relu_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar, int64_t, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 4))).toScalar(),
          (std::move(peek(*stack, 2, 4))).toInt(),
          (std::move(peek(*stack, 3, 4))).toScalar());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::replication_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toIntVector());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::replication_pad2d_backward.grad_input(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toIntVector());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, c10::optional<MemoryFormat>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toOptional<c10::MemoryFormat>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::result_type.Tensor(Tensor tensor, Tensor other) -> ScalarType",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<ScalarType, const Tensor &, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::result_type.Scalar_Tensor(Scalar scalar, Tensor tensor) -> ScalarType",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<ScalarType, Scalar, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toScalar(),
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::result_type.Scalar(Tensor tensor, Scalar other) -> ScalarType",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<ScalarType, const Tensor &, Scalar>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::result_type.Scalar_Scalar(Scalar scalar1, Scalar scalar2) -> ScalarType",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<ScalarType, Scalar, Scalar>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toScalar(),
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::rrelu_(Tensor(a!) self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 5))).toScalar(),
          (std::move(peek(*stack, 2, 5))).toScalar(),
          (std::move(peek(*stack, 3, 5))).toBool(),
          (std::move(peek(*stack, 4, 5))).toOptional<at::Generator>());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toScalar(),
          (std::move(peek(*stack, 3, 6))).toScalar(),
          (std::move(peek(*stack, 4, 6))).toBool(),
          (std::move(peek(*stack, 5, 6))).toOptional<at::Generator>());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 4))).toInt(),
          (std::move(peek(*stack, 2, 4))).toTensor(),
          (std::move(peek(*stack, 3, 4))).toTensor());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 4))).toInt(),
          (std::move(peek(*stack, 2, 4))).toTensor(),
          (std::move(peek(*stack, 3, 4))).toScalar());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::searchsorted.Tensor_out(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, bool, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toBool(),
          (std::move(peek(*stack, 3, 5))).toBool());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::set_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::sinh_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::slow_conv3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 6, 7))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          toOptionalTensor((std::move(peek(*stack, 3, 7)))),
          (std::move(peek(*stack, 4, 7))).toIntVector(),
          (std::move(peek(*stack, 5, 7))).toIntVector());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::slow_conv_transpose3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 8, 9))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 9))).toTensor(),
          (std::move(peek(*stack, 1, 9))).toTensor(),
          (std::move(peek(*stack, 2, 9))).toIntVector(),
          toOptionalTensor((std::move(peek(*stack, 3, 9)))),
          (std::move(peek(*stack, 4, 9))).toIntVector(),
          (std::move(peek(*stack, 5, 9))).toIntVector(),
          (std::move(peek(*stack, 6, 9))).toIntVector(),
          (std::move(peek(*stack, 7, 9))).toIntVector());
          drop(*stack, 9);
          pack(*stack, std::move(result_));
      })
    .op("aten::smooth_l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toTensor(),
          (std::move(peek(*stack, 3, 5))).toInt());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::softshrink_backward.grad_input(Tensor grad_output, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toScalar());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 4))).toIntVector(),
          (std::move(peek(*stack, 2, 4))).toInt(),
          (std::move(peek(*stack, 3, 4))).toInt());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::sqrt_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::squeeze_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toInt());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, TensorList, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensorVector(),
          (std::move(peek(*stack, 1, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toScalar());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::t_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::tan_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toIntVector(),
          toOptionalTensor((std::move(peek(*stack, 3, 6)))),
          (std::move(peek(*stack, 4, 6))).toIntVector(),
          (std::move(peek(*stack, 5, 6))).toIntVector());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::thnn_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toIntVector(),
          toOptionalTensor((std::move(peek(*stack, 3, 6)))),
          (std::move(peek(*stack, 4, 6))).toIntVector(),
          (std::move(peek(*stack, 5, 6))).toIntVector());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::thnn_conv_depthwise2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(unboxedKernel, (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          toOptionalTensor((std::move(peek(*stack, 3, 7)))),
          (std::move(peek(*stack, 4, 7))).toIntVector(),
          (std::move(peek(*stack, 5, 7))).toIntVector(),
          (std::move(peek(*stack, 6, 7))).toIntVector());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::thnn_conv_depthwise2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(unboxedKernel, (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          toOptionalTensor((std::move(peek(*stack, 3, 7)))),
          (std::move(peek(*stack, 4, 7))).toIntVector(),
          (std::move(peek(*stack, 5, 7))).toIntVector(),
          (std::move(peek(*stack, 6, 7))).toIntVector());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toInt());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 4, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 5, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::tril_indices((std::move(peek(*stack, 0, 7))).toInt(),
          (std::move(peek(*stack, 1, 7))).toInt(),
          (std::move(peek(*stack, 2, 7))).toInt(),
          options);
          #else
              auto result_ = torch::tril_indices((std::move(peek(*stack, 0, 7))).toInt(),
          (std::move(peek(*stack, 1, 7))).toInt(),
          (std::move(peek(*stack, 2, 7))).toInt(),
          options);
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toInt());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 4, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 5, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::triu_indices((std::move(peek(*stack, 0, 7))).toInt(),
          (std::move(peek(*stack, 1, 7))).toInt(),
          (std::move(peek(*stack, 2, 7))).toInt(),
          options);
          #else
              auto result_ = torch::triu_indices((std::move(peek(*stack, 0, 7))).toInt(),
          (std::move(peek(*stack, 1, 7))).toInt(),
          (std::move(peek(*stack, 2, 7))).toInt(),
          options);
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::true_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_bicubic2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 6, 7))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toIntVector(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          (std::move(peek(*stack, 3, 7))).toBool(),
          (std::move(peek(*stack, 4, 7))).toOptional<double>(),
          (std::move(peek(*stack, 5, 7))).toOptional<double>());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_bilinear2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 6, 7))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toIntVector(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          (std::move(peek(*stack, 3, 7))).toBool(),
          (std::move(peek(*stack, 4, 7))).toOptional<double>(),
          (std::move(peek(*stack, 5, 7))).toOptional<double>());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_linear1d_backward.grad_input(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, float? scales=None, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toIntVector(),
          (std::move(peek(*stack, 2, 6))).toIntVector(),
          (std::move(peek(*stack, 3, 6))).toBool(),
          (std::move(peek(*stack, 4, 6))).toOptional<double>());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_nearest1d.out(Tensor self, int[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, c10::optional<double>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toIntVector(),
          (std::move(peek(*stack, 2, 4))).toOptional<double>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toIntVector(),
          (std::move(peek(*stack, 2, 6))).toIntVector(),
          (std::move(peek(*stack, 3, 6))).toOptional<double>(),
          (std::move(peek(*stack, 4, 6))).toOptional<double>());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
  ;

} // anon namespace


}} // namespace torch::jit
