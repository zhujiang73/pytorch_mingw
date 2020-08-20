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
    .op("aten::__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::__ixor__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::__ixor__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::_amp_update_scale(Tensor(a!) growth_tracker, Tensor current_scale, Tensor found_inf, float scale_growth_factor, float scale_backoff_factor, int growth_interval) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto growth_tracker = (std::move(peek(*stack, 0, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor, Tensor &, const Tensor &, const Tensor &, double, double, int64_t>(unboxedKernel, growth_tracker,
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          (std::move(peek(*stack, 3, 6))).toDouble(),
          (std::move(peek(*stack, 4, 6))).toDouble(),
          (std::move(peek(*stack, 5, 6))).toInt());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, bool>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toBool());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::_cudnn_init_dropout_state(float dropout, bool train, int dropout_seed, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toScalarType())
                  .layout((std::move(peek(*stack, 4, 7))).toLayout())
                  .device((std::move(peek(*stack, 5, 7))).toDevice())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toBool());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::_cudnn_init_dropout_state((std::move(peek(*stack, 0, 7))).toDouble(),
          (std::move(peek(*stack, 1, 7))).toBool(),
          (std::move(peek(*stack, 2, 7))).toInt(),
          options);
          #else
              auto result_ = torch::_cudnn_init_dropout_state((std::move(peek(*stack, 0, 7))).toDouble(),
          (std::move(peek(*stack, 1, 7))).toBool(),
          (std::move(peek(*stack, 2, 7))).toInt(),
          options);
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &, bool>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          (std::move(peek(*stack, 2, 8))).toTensor(),
          (std::move(peek(*stack, 3, 8))).toBool(),
          (std::move(peek(*stack, 4, 8))).toInt(),
          (std::move(peek(*stack, 5, 8))).toBool(),
          toOptionalTensor((std::move(peek(*stack, 6, 8)))),
          (std::move(peek(*stack, 7, 8))).toBool());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 10))).toTensor(),
          (std::move(peek(*stack, 1, 10))).toTensor(),
          (std::move(peek(*stack, 2, 10))).toTensor(),
          (std::move(peek(*stack, 3, 10))).toTensor(),
          (std::move(peek(*stack, 4, 10))).toTensor(),
          (std::move(peek(*stack, 5, 10))).toTensor(),
          (std::move(peek(*stack, 6, 10))).toInt(),
          (std::move(peek(*stack, 7, 10))).toBool(),
          (std::move(peek(*stack, 8, 10))).toInt(),
          toOptionalTensor((std::move(peek(*stack, 9, 10)))));
          drop(*stack, 10);
          pack(*stack, std::move(result_));
      })
    .op("aten::_multinomial_alias_draw(Tensor J, Tensor q, int num_samples, *, Generator? generator=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, int64_t, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toInt(),
          (std::move(peek(*stack, 3, 4))).toOptional<at::Generator>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sample_dirichlet(Tensor self, Generator? generator=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          (std::move(peek(*stack, 1, 2))).toOptional<at::Generator>());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sobol_engine_ff_(Tensor(a!) self, int n, Tensor sobolstate, int dimension, int num_generated) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, const Tensor &, int64_t, int64_t>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 5))).toInt(),
          (std::move(peek(*stack, 2, 5))).toTensor(),
          (std::move(peek(*stack, 3, 5))).toInt(),
          (std::move(peek(*stack, 4, 5))).toInt());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sparse_coo_tensor_with_dims(int sparse_dim, int dense_dim, int[] size, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toScalarType())
                  .layout((std::move(peek(*stack, 4, 7))).toLayout())
                  .device((std::move(peek(*stack, 5, 7))).toDevice())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toBool());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::_sparse_coo_tensor_with_dims((std::move(peek(*stack, 0, 7))).toInt(),
          (std::move(peek(*stack, 1, 7))).toInt(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          options);
          #else
              auto result_ = torch::_sparse_coo_tensor_with_dims((std::move(peek(*stack, 0, 7))).toInt(),
          (std::move(peek(*stack, 1, 7))).toInt(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          options);
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::_thnn_differentiable_gru_cell_backward(Tensor grad_hy, Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias, Tensor? hidden_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          (std::move(peek(*stack, 3, 6))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 4, 6)))),
          toOptionalTensor((std::move(peek(*stack, 5, 6)))));
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::_thnn_differentiable_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor input_gates, Tensor hidden_gates, Tensor? input_bias, Tensor? hidden_bias, Tensor cx, Tensor cy) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, toOptionalTensor((std::move(peek(*stack, 0, 8)))),
          toOptionalTensor((std::move(peek(*stack, 1, 8)))),
          (std::move(peek(*stack, 2, 8))).toTensor(),
          (std::move(peek(*stack, 3, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 4, 8)))),
          toOptionalTensor((std::move(peek(*stack, 5, 8)))),
          (std::move(peek(*stack, 6, 8))).toTensor(),
          (std::move(peek(*stack, 7, 8))).toTensor());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::_thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 5)))),
          toOptionalTensor((std::move(peek(*stack, 4, 5)))));
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 5)))),
          toOptionalTensor((std::move(peek(*stack, 4, 5)))));
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::acosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::adaptive_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)",
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
    .op("aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toScalar(),
          (std::move(peek(*stack, 2, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toTensor(),
          (std::move(peek(*stack, 3, 5))).toScalar());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
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
    .op("aten::all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toInt(),
          (std::move(peek(*stack, 2, 4))).toBool());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toInt(),
          (std::move(peek(*stack, 2, 4))).toBool());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::asinh_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::atan_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::avg_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
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
    .op("aten::bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 5))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::bartlett_window((std::move(peek(*stack, 0, 5))).toInt(),
          options);
          #else
              auto result_ = torch::bartlett_window((std::move(peek(*stack, 0, 5))).toInt(),
          options);
          #endif
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::bartlett_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::bartlett_window((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toBool(),
          options);
          #else
              auto result_ = torch::bartlett_window((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toBool(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor mean_dy, Tensor mean_dy_xmu) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toTensor(),
          (std::move(peek(*stack, 3, 7))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 4, 7)))),
          (std::move(peek(*stack, 5, 7))).toTensor(),
          (std::move(peek(*stack, 6, 7))).toTensor());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, c10::optional<Generator>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, double, c10::optional<Generator>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toDouble(),
          (std::move(peek(*stack, 2, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 4)))),
          (std::move(peek(*stack, 3, 4))).toInt());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::binary_cross_entropy_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 6)))),
          (std::move(peek(*stack, 4, 6))).toInt());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 1, 3)))),
          (std::move(peek(*stack, 2, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_and_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_and_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toOptional<Scalar>(),
          (std::move(peek(*stack, 2, 3))).toOptional<Scalar>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor",
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
    .op("aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor",
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
    .op("aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
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
    .op("aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::cosh_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::cudnn_convolution.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(unboxedKernel, (std::move(peek(*stack, 0, 9))).toTensor(),
          (std::move(peek(*stack, 1, 9))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 9)))),
          (std::move(peek(*stack, 3, 9))).toIntVector(),
          (std::move(peek(*stack, 4, 9))).toIntVector(),
          (std::move(peek(*stack, 5, 9))).toIntVector(),
          (std::move(peek(*stack, 6, 9))).toInt(),
          (std::move(peek(*stack, 7, 9))).toBool(),
          (std::move(peek(*stack, 8, 9))).toBool());
          drop(*stack, 9);
          pack(*stack, std::move(result_));
      })
    .op("aten::cudnn_convolution_transpose.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(unboxedKernel, (std::move(peek(*stack, 0, 10))).toTensor(),
          (std::move(peek(*stack, 1, 10))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 10)))),
          (std::move(peek(*stack, 3, 10))).toIntVector(),
          (std::move(peek(*stack, 4, 10))).toIntVector(),
          (std::move(peek(*stack, 5, 10))).toIntVector(),
          (std::move(peek(*stack, 6, 10))).toIntVector(),
          (std::move(peek(*stack, 7, 10))).toInt(),
          (std::move(peek(*stack, 8, 10))).toBool(),
          (std::move(peek(*stack, 9, 10))).toBool());
          drop(*stack, 10);
          pack(*stack, std::move(result_));
      })
    .op("aten::digamma_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::elu_backward.grad_input(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar, Scalar, Scalar, const Tensor &>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toScalar(),
          (std::move(peek(*stack, 2, 6))).toScalar(),
          (std::move(peek(*stack, 3, 6))).toScalar(),
          (std::move(peek(*stack, 4, 6))).toTensor());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::empty.out(int[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, IntArrayRef, c10::optional<MemoryFormat>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toIntVector(),
          (std::move(peek(*stack, 1, 3))).toOptional<c10::MemoryFormat>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::exp_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, double, c10::optional<Generator>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toDouble(),
          (std::move(peek(*stack, 2, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::eye(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 5))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::eye((std::move(peek(*stack, 0, 5))).toInt(),
          options);
          #else
              auto result_ = torch::eye((std::move(peek(*stack, 0, 5))).toInt(),
          options);
          #endif
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::eye.m(int n, int m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::eye((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toInt(),
          options);
          #else
              auto result_ = torch::eye((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toInt(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::floor_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::fractional_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)",
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
    .op("aten::from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 4, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 5, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::from_file((std::move(peek(*stack, 0, 7))).toStringRef(),
          (std::move(peek(*stack, 1, 7))).toOptional<bool>(),
          (std::move(peek(*stack, 2, 7))).toOptional<int64_t>(),
          options);
          #else
              auto result_ = torch::from_file((std::move(peek(*stack, 0, 7))).toStringRef(),
          (std::move(peek(*stack, 1, 7))).toOptional<bool>(),
          (std::move(peek(*stack, 2, 7))).toOptional<int64_t>(),
          options);
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::glu.out(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          (std::move(peek(*stack, 3, 6))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 4, 6)))),
          toOptionalTensor((std::move(peek(*stack, 5, 6)))));
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 5))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::hamming_window((std::move(peek(*stack, 0, 5))).toInt(),
          options);
          #else
              auto result_ = torch::hamming_window((std::move(peek(*stack, 0, 5))).toInt(),
          options);
          #endif
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::hamming_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::hamming_window((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toBool(),
          options);
          #else
              auto result_ = torch::hamming_window((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toBool(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::hamming_window.periodic_alpha(int window_length, bool periodic, float alpha, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 4, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 5, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::hamming_window((std::move(peek(*stack, 0, 7))).toInt(),
          (std::move(peek(*stack, 1, 7))).toBool(),
          (std::move(peek(*stack, 2, 7))).toDouble(),
          options);
          #else
              auto result_ = torch::hamming_window((std::move(peek(*stack, 0, 7))).toInt(),
          (std::move(peek(*stack, 1, 7))).toBool(),
          (std::move(peek(*stack, 2, 7))).toDouble(),
          options);
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::hamming_window.periodic_alpha_beta(int window_length, bool periodic, float alpha, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 4, 8))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 5, 8))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 6, 8))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 7, 8))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::hamming_window((std::move(peek(*stack, 0, 8))).toInt(),
          (std::move(peek(*stack, 1, 8))).toBool(),
          (std::move(peek(*stack, 2, 8))).toDouble(),
          (std::move(peek(*stack, 3, 8))).toDouble(),
          options);
          #else
              auto result_ = torch::hamming_window((std::move(peek(*stack, 0, 8))).toInt(),
          (std::move(peek(*stack, 1, 8))).toBool(),
          (std::move(peek(*stack, 2, 8))).toDouble(),
          (std::move(peek(*stack, 3, 8))).toDouble(),
          options);
          #endif
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::hardswish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::hardtanh_backward.grad_input(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toScalar(),
          (std::move(peek(*stack, 3, 5))).toScalar());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, TensorList>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          toListOfOptionalTensor((std::move(peek(*stack, 1, 2)))));
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)",
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
    .op("aten::istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool onesided=True, int? length=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool, bool, c10::optional<int64_t>>(unboxedKernel, (std::move(peek(*stack, 0, 9))).toTensor(),
          (std::move(peek(*stack, 1, 9))).toInt(),
          (std::move(peek(*stack, 2, 9))).toOptional<int64_t>(),
          (std::move(peek(*stack, 3, 9))).toOptional<int64_t>(),
          toOptionalTensor((std::move(peek(*stack, 4, 9)))),
          (std::move(peek(*stack, 5, 9))).toBool(),
          (std::move(peek(*stack, 6, 9))).toBool(),
          (std::move(peek(*stack, 7, 9))).toBool(),
          (std::move(peek(*stack, 8, 9))).toOptional<int64_t>());
          drop(*stack, 9);
          pack(*stack, std::move(result_));
      })
    .op("aten::l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)",
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
    .op("aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 3)))));
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::linspace.out(Scalar start, Scalar end, int steps=100, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar, Scalar, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toScalar(),
          (std::move(peek(*stack, 1, 4))).toScalar(),
          (std::move(peek(*stack, 2, 4))).toInt());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::log1p_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::log2_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::log_sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt(),
          (std::move(peek(*stack, 2, 3))).toOptional<ScalarType>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::logical_or.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor>, const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensorVector(),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          (std::move(peek(*stack, 3, 6))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 4, 6)))),
          toOptionalTensor((std::move(peek(*stack, 5, 6)))));
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::max_unpool2d.out(Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toIntVector());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::max_unpool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 6, 7))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toTensor(),
          (std::move(peek(*stack, 3, 7))).toIntVector(),
          (std::move(peek(*stack, 4, 7))).toIntVector(),
          (std::move(peek(*stack, 5, 7))).toIntVector());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
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
    .op("aten::miopen_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>>, const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,4>>(unboxedKernel, (std::move(peek(*stack, 0, 21))).toTensor(),
          (std::move(peek(*stack, 1, 21))).toTensorVector(),
          (std::move(peek(*stack, 2, 21))).toInt(),
          (std::move(peek(*stack, 3, 21))).toTensor(),
          (std::move(peek(*stack, 4, 21))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 5, 21)))),
          (std::move(peek(*stack, 6, 21))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 7, 21)))),
          toOptionalTensor((std::move(peek(*stack, 8, 21)))),
          toOptionalTensor((std::move(peek(*stack, 9, 21)))),
          (std::move(peek(*stack, 10, 21))).toInt(),
          (std::move(peek(*stack, 11, 21))).toInt(),
          (std::move(peek(*stack, 12, 21))).toInt(),
          (std::move(peek(*stack, 13, 21))).toBool(),
          (std::move(peek(*stack, 14, 21))).toDouble(),
          (std::move(peek(*stack, 15, 21))).toBool(),
          (std::move(peek(*stack, 16, 21))).toBool(),
          (std::move(peek(*stack, 17, 21))).toIntVector(),
          toOptionalTensor((std::move(peek(*stack, 18, 21)))),
          (std::move(peek(*stack, 19, 21))).toTensor(),
          as_bool_array<4>((std::move(peek(*stack, 20, 21))).toBoolList()));
          drop(*stack, 21);
          pack(*stack, std::move(result_));
      })
    .op("aten::mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups) -> Tensor",
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
    .op("aten::mse_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)",
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
    .op("aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toScalar(),
          (std::move(peek(*stack, 3, 6))).toScalar(),
          toOptionalTensor((std::move(peek(*stack, 4, 6)))),
          (std::move(peek(*stack, 5, 6))).toInt());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::multi_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 7, 8))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          (std::move(peek(*stack, 2, 8))).toTensor(),
          (std::move(peek(*stack, 3, 8))).toScalar(),
          (std::move(peek(*stack, 4, 8))).toScalar(),
          toOptionalTensor((std::move(peek(*stack, 5, 8)))),
          (std::move(peek(*stack, 6, 8))).toInt());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::multilabel_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toInt());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t, bool, c10::optional<Generator>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toInt(),
          (std::move(peek(*stack, 2, 5))).toBool(),
          (std::move(peek(*stack, 3, 5))).toOptional<at::Generator>());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, int N, int C, int HxW, int group, float eps) -> (Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, int64_t, double>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 1, 8)))),
          toOptionalTensor((std::move(peek(*stack, 2, 8)))),
          (std::move(peek(*stack, 3, 8))).toInt(),
          (std::move(peek(*stack, 4, 8))).toInt(),
          (std::move(peek(*stack, 5, 8))).toInt(),
          (std::move(peek(*stack, 6, 8))).toInt(),
          (std::move(peek(*stack, 7, 8))).toDouble());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::new_full(Tensor self, int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 4, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 5, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toOptional<bool>());
          auto result_ = ((std::move(peek(*stack, 0, 7))).toTensor()).new_full((std::move(peek(*stack, 1, 7))).toIntVector(),
          (std::move(peek(*stack, 2, 7))).toScalar(),
          options);
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 7)))),
          (std::move(peek(*stack, 4, 7))).toInt(),
          (std::move(peek(*stack, 5, 7))).toInt(),
          (std::move(peek(*stack, 6, 7))).toTensor());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 7)))),
          (std::move(peek(*stack, 4, 7))).toInt(),
          (std::move(peek(*stack, 5, 7))).toInt(),
          (std::move(peek(*stack, 6, 7))).toTensor());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, c10::optional<Generator>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toOptional<at::Generator>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, double, const Tensor &, c10::optional<Generator>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toDouble(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toOptional<at::Generator>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::normal.Tensor_float_out(Tensor mean, float std=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, double, c10::optional<Generator>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toDouble(),
          (std::move(peek(*stack, 2, 4))).toOptional<at::Generator>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::normal.float_float_out(float mean, float std, int[] size, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, double, double, IntArrayRef, c10::optional<Generator>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toDouble(),
          (std::move(peek(*stack, 1, 5))).toDouble(),
          (std::move(peek(*stack, 2, 5))).toIntVector(),
          (std::move(peek(*stack, 3, 5))).toOptional<at::Generator>());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::nuclear_norm.out(Tensor self, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toBool());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::nuclear_norm.dim_out(Tensor self, int[2] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 5))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::ones((std::move(peek(*stack, 0, 5))).toIntVector(),
          options);
          #else
              auto result_ = torch::ones((std::move(peek(*stack, 0, 5))).toIntVector(),
          options);
          #endif
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::ones_like((std::move(peek(*stack, 0, 6))).toTensor(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #else
              auto result_ = torch::ones_like((std::move(peek(*stack, 0, 6))).toTensor(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::poisson(Tensor self, Generator? generator=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          (std::move(peek(*stack, 1, 2))).toOptional<at::Generator>());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toInt(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, c10::optional<ScalarType>>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          (std::move(peek(*stack, 1, 2))).toOptional<ScalarType>());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, int64_t, bool, c10::optional<ScalarType>>(unboxedKernel, (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toInt(),
          (std::move(peek(*stack, 2, 4))).toBool(),
          (std::move(peek(*stack, 3, 4))).toOptional<ScalarType>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toTensor(),
          (std::move(peek(*stack, 3, 4))).toBool());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::quantized_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 1, 8)))),
          toOptionalTensor((std::move(peek(*stack, 2, 8)))),
          (std::move(peek(*stack, 3, 8))).toTensor(),
          (std::move(peek(*stack, 4, 8))).toTensor(),
          (std::move(peek(*stack, 5, 8))).toDouble(),
          (std::move(peek(*stack, 6, 8))).toDouble(),
          (std::move(peek(*stack, 7, 8))).toInt());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::rand.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toIntVector());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::rand.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, IntArrayRef, c10::optional<Generator>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toIntVector(),
          (std::move(peek(*stack, 1, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::randint(int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::randint((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toIntVector(),
          options);
          #else
              auto result_ = torch::randint((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toIntVector(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::randint.generator(int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 4, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 5, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::randint((std::move(peek(*stack, 0, 7))).toInt(),
          (std::move(peek(*stack, 1, 7))).toIntVector(),
          (std::move(peek(*stack, 2, 7))).toOptional<at::Generator>(),
          options);
          #else
              auto result_ = torch::randint((std::move(peek(*stack, 0, 7))).toInt(),
          (std::move(peek(*stack, 1, 7))).toIntVector(),
          (std::move(peek(*stack, 2, 7))).toOptional<at::Generator>(),
          options);
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::randint.low(int low, int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 4, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 5, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::randint((std::move(peek(*stack, 0, 7))).toInt(),
          (std::move(peek(*stack, 1, 7))).toInt(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          options);
          #else
              auto result_ = torch::randint((std::move(peek(*stack, 0, 7))).toInt(),
          (std::move(peek(*stack, 1, 7))).toInt(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          options);
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::randint.low_generator(int low, int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 4, 8))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 5, 8))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 6, 8))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 7, 8))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::randint((std::move(peek(*stack, 0, 8))).toInt(),
          (std::move(peek(*stack, 1, 8))).toInt(),
          (std::move(peek(*stack, 2, 8))).toIntVector(),
          (std::move(peek(*stack, 3, 8))).toOptional<at::Generator>(),
          options);
          #else
              auto result_ = torch::randint((std::move(peek(*stack, 0, 8))).toInt(),
          (std::move(peek(*stack, 1, 8))).toInt(),
          (std::move(peek(*stack, 2, 8))).toIntVector(),
          (std::move(peek(*stack, 3, 8))).toOptional<at::Generator>(),
          options);
          #endif
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::randint_like(Tensor self, int high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 7))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::randint_like((std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toInt(),
          options,
          (std::move(peek(*stack, 6, 7))).toOptional<c10::MemoryFormat>());
          #else
              auto result_ = torch::randint_like((std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toInt(),
          options,
          (std::move(peek(*stack, 6, 7))).toOptional<c10::MemoryFormat>());
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::randint_like.low_dtype(Tensor self, int low, int high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 8))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 4, 8))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 5, 8))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 6, 8))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::randint_like((std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toInt(),
          (std::move(peek(*stack, 2, 8))).toInt(),
          options,
          (std::move(peek(*stack, 7, 8))).toOptional<c10::MemoryFormat>());
          #else
              auto result_ = torch::randint_like((std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toInt(),
          (std::move(peek(*stack, 2, 8))).toInt(),
          options,
          (std::move(peek(*stack, 7, 8))).toOptional<c10::MemoryFormat>());
          #endif
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::randperm(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 5))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::randperm((std::move(peek(*stack, 0, 5))).toInt(),
          options);
          #else
              auto result_ = torch::randperm((std::move(peek(*stack, 0, 5))).toInt(),
          options);
          #endif
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::randperm.generator(int n, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::randperm((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toOptional<at::Generator>(),
          options);
          #else
              auto result_ = torch::randperm((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toOptional<at::Generator>(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::reflection_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toIntVector());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::replication_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toIntVector());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::replication_pad3d_backward.grad_input(Tensor grad_output, Tensor self, int[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)",
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
    .op("aten::requires_grad_(Tensor(a!) self, bool requires_grad=True) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, bool>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toBool());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          (std::move(peek(*stack, 3, 6))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 4, 6)))),
          toOptionalTensor((std::move(peek(*stack, 5, 6)))));
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::round_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toScalar(),
          (std::move(peek(*stack, 2, 5))).toScalar(),
          (std::move(peek(*stack, 3, 5))).toBool(),
          (std::move(peek(*stack, 4, 5))).toOptional<at::Generator>());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toScalar(),
          (std::move(peek(*stack, 3, 6))).toScalar(),
          (std::move(peek(*stack, 4, 6))).toBool(),
          (std::move(peek(*stack, 5, 6))).toOptional<at::Generator>());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::selu_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::sin_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::slow_conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor",
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
    .op("aten::slow_conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          (std::move(peek(*stack, 2, 8))).toIntVector(),
          toOptionalTensor((std::move(peek(*stack, 3, 8)))),
          (std::move(peek(*stack, 4, 8))).toIntVector(),
          (std::move(peek(*stack, 5, 8))).toIntVector(),
          (std::move(peek(*stack, 6, 8))).toIntVector(),
          (std::move(peek(*stack, 7, 8))).toIntVector());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::smooth_l1_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toInt());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::soft_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)",
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
    .op("aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt(),
          (std::move(peek(*stack, 2, 3))).toOptional<ScalarType>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::softplus_backward.grad_input(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toScalar(),
          (std::move(peek(*stack, 3, 6))).toScalar(),
          (std::move(peek(*stack, 4, 6))).toTensor());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::sparse_coo_tensor.size(int[] size, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 5))).toScalarType())
                  .layout((std::move(peek(*stack, 2, 5))).toLayout())
                  .device((std::move(peek(*stack, 3, 5))).toDevice())
                  .pinned_memory((std::move(peek(*stack, 4, 5))).toBool());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::sparse_coo_tensor((std::move(peek(*stack, 0, 5))).toIntVector(),
          options);
          #else
              auto result_ = torch::sparse_coo_tensor((std::move(peek(*stack, 0, 5))).toIntVector(),
          options);
          #endif
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::sparse_coo_tensor.indices(Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::sparse_coo_tensor((std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          options);
          #else
              auto result_ = torch::sparse_coo_tensor((std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::sparse_coo_tensor.indices_size(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 4, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 5, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::sparse_coo_tensor((std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          options);
          #else
              auto result_ = torch::sparse_coo_tensor((std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          options);
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)",
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
    .op("aten::square_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::std.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toIntVector(),
          (std::move(peek(*stack, 2, 5))).toBool(),
          (std::move(peek(*stack, 3, 5))).toBool());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool>(unboxedKernel, (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toInt(),
          (std::move(peek(*stack, 2, 7))).toOptional<int64_t>(),
          (std::move(peek(*stack, 3, 7))).toOptional<int64_t>(),
          toOptionalTensor((std::move(peek(*stack, 4, 7)))),
          (std::move(peek(*stack, 5, 7))).toBool(),
          (std::move(peek(*stack, 6, 7))).toBool());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toScalar(),
          (std::move(peek(*stack, 2, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, c10::optional<ScalarType>>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          (std::move(peek(*stack, 1, 2))).toOptional<ScalarType>());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>>(unboxedKernel, (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toIntVector(),
          (std::move(peek(*stack, 2, 4))).toBool(),
          (std::move(peek(*stack, 3, 4))).toOptional<ScalarType>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::thnn_conv2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, *, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::thnn_conv_depthwise2d_forward.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 7, 8))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          (std::move(peek(*stack, 2, 8))).toIntVector(),
          toOptionalTensor((std::move(peek(*stack, 3, 8)))),
          (std::move(peek(*stack, 4, 8))).toIntVector(),
          (std::move(peek(*stack, 5, 8))).toIntVector(),
          (std::move(peek(*stack, 6, 8))).toIntVector());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::thnn_conv_depthwise2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 7, 8))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          (std::move(peek(*stack, 2, 8))).toIntVector(),
          toOptionalTensor((std::move(peek(*stack, 3, 8)))),
          (std::move(peek(*stack, 4, 8))).toIntVector(),
          (std::move(peek(*stack, 5, 8))).toIntVector(),
          (std::move(peek(*stack, 6, 8))).toIntVector());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toScalar(),
          (std::move(peek(*stack, 2, 4))).toScalar());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::to.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, bool, bool, c10::optional<MemoryFormat>>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toBool(),
          (std::move(peek(*stack, 3, 5))).toBool(),
          (std::move(peek(*stack, 4, 5))).toOptional<c10::MemoryFormat>());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, ScalarType, bool, bool, c10::optional<MemoryFormat>>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toScalarType(),
          (std::move(peek(*stack, 2, 5))).toBool(),
          (std::move(peek(*stack, 3, 5))).toBool(),
          (std::move(peek(*stack, 4, 5))).toOptional<c10::MemoryFormat>());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, Device, ScalarType, bool, bool, c10::optional<MemoryFormat>>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toDevice(),
          (std::move(peek(*stack, 2, 6))).toScalarType(),
          (std::move(peek(*stack, 3, 6))).toBool(),
          (std::move(peek(*stack, 4, 6))).toBool(),
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::to.dtype_layout(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 8))).toScalarType())
                  .layout((std::move(peek(*stack, 2, 8))).toLayout())
                  .device((std::move(peek(*stack, 3, 8))).toDevice())
                  .pinned_memory((std::move(peek(*stack, 4, 8))).toBool());
          auto result_ = ((std::move(peek(*stack, 0, 8))).toTensor()).to(options,
          (std::move(peek(*stack, 5, 8))).toBool(),
          (std::move(peek(*stack, 6, 8))).toBool(),
          (std::move(peek(*stack, 7, 8))).toOptional<c10::MemoryFormat>());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::unfold_backward(Tensor grad_in, int[] input_sizes, int dim, int size, int step) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, IntArrayRef, int64_t, int64_t, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toIntVector(),
          (std::move(peek(*stack, 2, 5))).toInt(),
          (std::move(peek(*stack, 3, 5))).toInt(),
          (std::move(peek(*stack, 4, 5))).toInt());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, double, double, c10::optional<Generator>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 4))).toDouble(),
          (std::move(peek(*stack, 2, 4))).toDouble(),
          (std::move(peek(*stack, 3, 4))).toOptional<at::Generator>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_bicubic2d.out(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toIntVector(),
          (std::move(peek(*stack, 2, 6))).toBool(),
          (std::move(peek(*stack, 3, 6))).toOptional<double>(),
          (std::move(peek(*stack, 4, 6))).toOptional<double>());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_bilinear2d.out(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toIntVector(),
          (std::move(peek(*stack, 2, 6))).toBool(),
          (std::move(peek(*stack, 3, 6))).toOptional<double>(),
          (std::move(peek(*stack, 4, 6))).toOptional<double>());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_linear1d.out(Tensor self, int[1] output_size, bool align_corners, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toIntVector(),
          (std::move(peek(*stack, 2, 5))).toBool(),
          (std::move(peek(*stack, 3, 5))).toOptional<double>());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_nearest2d.out(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toIntVector(),
          (std::move(peek(*stack, 2, 5))).toOptional<double>(),
          (std::move(peek(*stack, 3, 5))).toOptional<double>());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_nearest3d_backward.grad_input(Tensor grad_output, int[3] output_size, int[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 6, 7))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toIntVector(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          (std::move(peek(*stack, 3, 7))).toOptional<double>(),
          (std::move(peek(*stack, 4, 7))).toOptional<double>(),
          (std::move(peek(*stack, 5, 7))).toOptional<double>());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_trilinear3d_backward.grad_input(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 7, 8))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toIntVector(),
          (std::move(peek(*stack, 2, 8))).toIntVector(),
          (std::move(peek(*stack, 3, 8))).toBool(),
          (std::move(peek(*stack, 4, 8))).toOptional<double>(),
          (std::move(peek(*stack, 5, 8))).toOptional<double>(),
          (std::move(peek(*stack, 6, 8))).toOptional<double>());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 5))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::zeros((std::move(peek(*stack, 0, 5))).toIntVector(),
          options);
          #else
              auto result_ = torch::zeros((std::move(peek(*stack, 0, 5))).toIntVector(),
          options);
          #endif
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::zeros_like((std::move(peek(*stack, 0, 6))).toTensor(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #else
              auto result_ = torch::zeros_like((std::move(peek(*stack, 0, 6))).toTensor(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
  ;

} // anon namespace


}} // namespace torch::jit
