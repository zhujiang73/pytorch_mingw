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
    .op("aten::__iand__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::__iand__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::_addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::_baddbmm_mkl_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
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
    .op("aten::_batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask, Tensor reservedSpace) -> (Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor>, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 12))).toInt(),
          (std::move(peek(*stack, 1, 12))).toTensor(),
          (std::move(peek(*stack, 2, 12))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 12)))),
          toOptionalTensor((std::move(peek(*stack, 4, 12)))),
          toOptionalTensor((std::move(peek(*stack, 5, 12)))),
          toOptionalTensor((std::move(peek(*stack, 6, 12)))),
          toOptionalTensor((std::move(peek(*stack, 7, 12)))),
          (std::move(peek(*stack, 8, 12))).toBool(),
          (std::move(peek(*stack, 9, 12))).toDouble(),
          as_bool_array<3>((std::move(peek(*stack, 10, 12))).toBoolList()),
          (std::move(peek(*stack, 11, 12))).toTensor());
          drop(*stack, 12);
          pack(*stack, std::move(result_));
      })
    .op("aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, TensorList, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensorVector(),
          (std::move(peek(*stack, 1, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool>(unboxedKernel, (std::move(peek(*stack, 0, 12))).toTensor(),
          (std::move(peek(*stack, 1, 12))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 12)))),
          (std::move(peek(*stack, 3, 12))).toIntVector(),
          (std::move(peek(*stack, 4, 12))).toIntVector(),
          (std::move(peek(*stack, 5, 12))).toIntVector(),
          (std::move(peek(*stack, 6, 12))).toBool(),
          (std::move(peek(*stack, 7, 12))).toIntVector(),
          (std::move(peek(*stack, 8, 12))).toInt(),
          (std::move(peek(*stack, 9, 12))).toBool(),
          (std::move(peek(*stack, 10, 12))).toBool(),
          (std::move(peek(*stack, 11, 12))).toBool());
          drop(*stack, 12);
          pack(*stack, std::move(result_));
      })
    .op("aten::_convolution_double_backward(Tensor? ggI, Tensor? ggW, Tensor? ggb, Tensor gO, Tensor weight, Tensor self, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool, std::array<bool,3>>(unboxedKernel, toOptionalTensor((std::move(peek(*stack, 0, 16)))),
          toOptionalTensor((std::move(peek(*stack, 1, 16)))),
          toOptionalTensor((std::move(peek(*stack, 2, 16)))),
          (std::move(peek(*stack, 3, 16))).toTensor(),
          (std::move(peek(*stack, 4, 16))).toTensor(),
          (std::move(peek(*stack, 5, 16))).toTensor(),
          (std::move(peek(*stack, 6, 16))).toIntVector(),
          (std::move(peek(*stack, 7, 16))).toIntVector(),
          (std::move(peek(*stack, 8, 16))).toIntVector(),
          (std::move(peek(*stack, 9, 16))).toBool(),
          (std::move(peek(*stack, 10, 16))).toIntVector(),
          (std::move(peek(*stack, 11, 16))).toInt(),
          (std::move(peek(*stack, 12, 16))).toBool(),
          (std::move(peek(*stack, 13, 16))).toBool(),
          (std::move(peek(*stack, 14, 16))).toBool(),
          as_bool_array<3>((std::move(peek(*stack, 15, 16))).toBoolList()));
          drop(*stack, 16);
          pack(*stack, std::move(result_));
      })
    .op("aten::_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 8)))),
          (std::move(peek(*stack, 3, 8))).toIntVector(),
          (std::move(peek(*stack, 4, 8))).toIntVector(),
          (std::move(peek(*stack, 5, 8))).toIntVector(),
          (std::move(peek(*stack, 6, 8))).toBool(),
          (std::move(peek(*stack, 7, 8))).toIntVector());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::_cudnn_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])",
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
    .op("aten::_cumprod.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::_cumsum.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::_embedding_bag_sparse_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 9))).toTensor(),
          (std::move(peek(*stack, 1, 9))).toTensor(),
          (std::move(peek(*stack, 2, 9))).toTensor(),
          (std::move(peek(*stack, 3, 9))).toTensor(),
          (std::move(peek(*stack, 4, 9))).toTensor(),
          (std::move(peek(*stack, 5, 9))).toInt(),
          (std::move(peek(*stack, 6, 9))).toBool(),
          (std::move(peek(*stack, 7, 9))).toInt(),
          toOptionalTensor((std::move(peek(*stack, 8, 9)))));
          drop(*stack, 9);
          pack(*stack, std::move(result_));
      })
    .op("aten::_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 8))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 8))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 8))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 8))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::_empty_affine_quantized((std::move(peek(*stack, 0, 8))).toIntVector(),
          options,
          (std::move(peek(*stack, 5, 8))).toDouble(),
          (std::move(peek(*stack, 6, 8))).toInt(),
          (std::move(peek(*stack, 7, 8))).toOptional<c10::MemoryFormat>());
          #else
              auto result_ = torch::_empty_affine_quantized((std::move(peek(*stack, 0, 8))).toIntVector(),
          options,
          (std::move(peek(*stack, 5, 8))).toDouble(),
          (std::move(peek(*stack, 6, 8))).toInt(),
          (std::move(peek(*stack, 7, 8))).toOptional<c10::MemoryFormat>());
          #endif
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor>, const Tensor &, double, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toDouble(),
          (std::move(peek(*stack, 2, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, TensorList, const Tensor &, bool, bool>(unboxedKernel, self,
          toListOfOptionalTensor((std::move(peek(*stack, 1, 5)))),
          (std::move(peek(*stack, 2, 5))).toTensor(),
          (std::move(peek(*stack, 3, 5))).toBool(),
          (std::move(peek(*stack, 4, 5))).toBool());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::_mkldnn_transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, int64_t>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toInt(),
          (std::move(peek(*stack, 2, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::_nnpack_spatial_convolution(Tensor input, Tensor weight, Tensor? bias, int[2] padding, int[2] stride=1) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 5)))),
          (std::move(peek(*stack, 3, 5))).toIntVector(),
          (std::move(peek(*stack, 4, 5))).toIntVector());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sobol_engine_initialize_state_(Tensor(a!) self, int dimension) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toInt());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sobol_engine_scramble_(Tensor(a!) self, Tensor ltm, int dimension) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sparse_coo_tensor_unsafe(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 4, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 5, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::_sparse_coo_tensor_unsafe((std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          options);
          #else
              auto result_ = torch::_sparse_coo_tensor_unsafe((std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          options);
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, int[] size, Tensor indices, Tensor values, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 5, 9))).toScalarType())
                  .layout((std::move(peek(*stack, 6, 9))).toLayout())
                  .device((std::move(peek(*stack, 7, 9))).toDevice())
                  .pinned_memory((std::move(peek(*stack, 8, 9))).toBool());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::_sparse_coo_tensor_with_dims_and_tensors((std::move(peek(*stack, 0, 9))).toInt(),
          (std::move(peek(*stack, 1, 9))).toInt(),
          (std::move(peek(*stack, 2, 9))).toIntVector(),
          (std::move(peek(*stack, 3, 9))).toTensor(),
          (std::move(peek(*stack, 4, 9))).toTensor(),
          options);
          #else
              auto result_ = torch::_sparse_coo_tensor_with_dims_and_tensors((std::move(peek(*stack, 0, 9))).toInt(),
          (std::move(peek(*stack, 1, 9))).toInt(),
          (std::move(peek(*stack, 2, 9))).toIntVector(),
          (std::move(peek(*stack, 3, 9))).toTensor(),
          (std::move(peek(*stack, 4, 9))).toTensor(),
          options);
          #endif
          drop(*stack, 9);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sparse_log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt(),
          (std::move(peek(*stack, 2, 3))).toOptional<ScalarType>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sparse_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, int64_t, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toInt(),
          (std::move(peek(*stack, 3, 4))).toTensor());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sparse_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt(),
          (std::move(peek(*stack, 2, 3))).toOptional<ScalarType>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sparse_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, int64_t, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toInt(),
          (std::move(peek(*stack, 3, 4))).toTensor());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sparse_sum.dtype(Tensor self, *, ScalarType dtype) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, ScalarType>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          (std::move(peek(*stack, 1, 2))).toScalarType());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::_sparse_sum.dim_dtype(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, IntArrayRef, ScalarType>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toIntVector(),
          (std::move(peek(*stack, 2, 3))).toScalarType());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::_standard_gamma(Tensor self, Generator? generator=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          (std::move(peek(*stack, 1, 2))).toOptional<at::Generator>());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::abs_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::acosh_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toIntVector());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::adaptive_avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
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
    .op("aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)",
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
    .op("aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::asin_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)",
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
    .op("aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool>(unboxedKernel, (std::move(peek(*stack, 0, 9))).toTensor(),
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
    .op("aten::batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool, bool>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          (std::move(peek(*stack, 2, 8))).toTensor(),
          (std::move(peek(*stack, 3, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 4, 8)))),
          (std::move(peek(*stack, 5, 8))).toBool(),
          (std::move(peek(*stack, 6, 8))).toBool(),
          (std::move(peek(*stack, 7, 8))).toBool());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 1, 6)))),
          toOptionalTensor((std::move(peek(*stack, 2, 6)))),
          (std::move(peek(*stack, 3, 6))).toTensor(),
          (std::move(peek(*stack, 4, 6))).toTensor(),
          (std::move(peek(*stack, 5, 6))).toDouble());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, double>(unboxedKernel, (std::move(peek(*stack, 0, 4))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 1, 4)))),
          toOptionalTensor((std::move(peek(*stack, 2, 4)))),
          (std::move(peek(*stack, 3, 4))).toDouble());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          (std::move(peek(*stack, 1, 2))).toOptional<at::Generator>());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, double, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toDouble(),
          (std::move(peek(*stack, 2, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 5)))),
          (std::move(peek(*stack, 3, 5))).toInt());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::binary_cross_entropy_with_logits_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 6)))),
          toOptionalTensor((std::move(peek(*stack, 4, 6)))),
          (std::move(peek(*stack, 5, 6))).toInt());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, c10::optional<Generator>>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_or.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_or.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_xor.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::bitwise_xor.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 5))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::blackman_window((std::move(peek(*stack, 0, 5))).toInt(),
          options);
          #else
              auto result_ = torch::blackman_window((std::move(peek(*stack, 0, 5))).toInt(),
          options);
          #endif
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::blackman_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::blackman_window((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toBool(),
          options);
          #else
              auto result_ = torch::blackman_window((std::move(peek(*stack, 0, 6))).toInt(),
          (std::move(peek(*stack, 1, 6))).toBool(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::can_cast(ScalarType from, ScalarType to) -> bool",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<bool, ScalarType, ScalarType>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toScalarType(),
          (std::move(peek(*stack, 1, 2))).toScalarType());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)",
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
    .op("aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toBool());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::cholesky_solve.out(Tensor self, Tensor input2, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::col2im_backward.grad_input(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toIntVector(),
          (std::move(peek(*stack, 2, 6))).toIntVector(),
          (std::move(peek(*stack, 3, 6))).toIntVector(),
          (std::move(peek(*stack, 4, 6))).toIntVector());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, MemoryFormat>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          (std::move(peek(*stack, 1, 2))).toMemoryFormat());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
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
    .op("aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor",
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
    .op("aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, bool>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toBool());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::cos_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::cross.out(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, c10::optional<int64_t>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toOptional<int64_t>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon, Tensor reserveSpace) -> (Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 9))).toTensor(),
          (std::move(peek(*stack, 1, 9))).toTensor(),
          (std::move(peek(*stack, 2, 9))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 9)))),
          toOptionalTensor((std::move(peek(*stack, 4, 9)))),
          toOptionalTensor((std::move(peek(*stack, 5, 9)))),
          toOptionalTensor((std::move(peek(*stack, 6, 9)))),
          (std::move(peek(*stack, 7, 9))).toDouble(),
          (std::move(peek(*stack, 8, 9))).toTensor());
          drop(*stack, 9);
          pack(*stack, std::move(result_));
      })
    .op("aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt(),
          (std::move(peek(*stack, 2, 3))).toOptional<ScalarType>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt(),
          (std::move(peek(*stack, 2, 3))).toOptional<ScalarType>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::deg2rad.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, double, bool>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toDouble(),
          (std::move(peek(*stack, 2, 3))).toBool());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar, Scalar, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toScalar(),
          (std::move(peek(*stack, 2, 5))).toScalar(),
          (std::move(peek(*stack, 3, 5))).toScalar());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)",
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
    .op("aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, double, double>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 4))).toTensor(),
          (std::move(peek(*stack, 2, 4))).toDouble(),
          (std::move(peek(*stack, 3, 4))).toDouble());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::empty_meta(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::empty_meta((std::move(peek(*stack, 0, 6))).toIntVector(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #else
              auto result_ = torch::empty_meta((std::move(peek(*stack, 0, 6))).toIntVector(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::empty_quantized(int[] size, Tensor qtensor) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, IntArrayRef, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toIntVector(),
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::eye.out(int n, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toInt());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::eye.m_out(int n, int m, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toInt(),
          (std::move(peek(*stack, 1, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::feature_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, double, bool>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toDouble(),
          (std::move(peek(*stack, 2, 3))).toBool());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::frac_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::full((std::move(peek(*stack, 0, 6))).toIntVector(),
          (std::move(peek(*stack, 1, 6))).toScalar(),
          options);
          #else
              auto result_ = torch::full((std::move(peek(*stack, 0, 6))).toIntVector(),
          (std::move(peek(*stack, 1, 6))).toScalar(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 7))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::full_like((std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toScalar(),
          options,
          (std::move(peek(*stack, 6, 7))).toOptional<c10::MemoryFormat>());
          #else
              auto result_ = torch::full_like((std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toScalar(),
          options,
          (std::move(peek(*stack, 6, 7))).toOptional<c10::MemoryFormat>());
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t, const Tensor &, bool>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toInt(),
          (std::move(peek(*stack, 2, 5))).toTensor(),
          (std::move(peek(*stack, 3, 5))).toBool());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::hardsigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::hardswish_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::histc.out(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t, Scalar, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toInt(),
          (std::move(peek(*stack, 2, 5))).toScalar(),
          (std::move(peek(*stack, 3, 5))).toScalar());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::im2col_backward.grad_input(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 6, 7))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toIntVector(),
          (std::move(peek(*stack, 2, 7))).toIntVector(),
          (std::move(peek(*stack, 3, 7))).toIntVector(),
          (std::move(peek(*stack, 4, 7))).toIntVector(),
          (std::move(peek(*stack, 5, 7))).toIntVector());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)",
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
    .op("aten::index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)",
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
    .op("aten::index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)",
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
    .op("aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, TensorList, const Tensor &, bool>(unboxedKernel, self,
          toListOfOptionalTensor((std::move(peek(*stack, 1, 4)))),
          (std::move(peek(*stack, 2, 4))).toTensor(),
          (std::move(peek(*stack, 3, 4))).toBool());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toInt(),
          (std::move(peek(*stack, 2, 4))).toTensor());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool>(unboxedKernel, (std::move(peek(*stack, 0, 9))).toTensor(),
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
    .op("aten::inverse.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::l1_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, IntArrayRef, const Tensor &, const Tensor &, double, bool>(unboxedKernel, (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toIntVector(),
          toOptionalTensor((std::move(peek(*stack, 2, 6)))),
          toOptionalTensor((std::move(peek(*stack, 3, 6)))),
          (std::move(peek(*stack, 4, 6))).toDouble(),
          (std::move(peek(*stack, 5, 6))).toBool());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::log_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)",
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
    .op("aten::logical_and.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::logical_or_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::logical_xor_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::logspace(Scalar start, Scalar end, int steps=100, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 4, 8))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 5, 8))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 6, 8))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 7, 8))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::logspace((std::move(peek(*stack, 0, 8))).toScalar(),
          (std::move(peek(*stack, 1, 8))).toScalar(),
          (std::move(peek(*stack, 2, 8))).toInt(),
          (std::move(peek(*stack, 3, 8))).toDouble(),
          options);
          #else
              auto result_ = torch::logspace((std::move(peek(*stack, 0, 8))).toScalar(),
          (std::move(peek(*stack, 1, 8))).toScalar(),
          (std::move(peek(*stack, 2, 8))).toInt(),
          (std::move(peek(*stack, 3, 8))).toDouble(),
          options);
          #endif
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toTensor(),
          (std::move(peek(*stack, 2, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::masked_select.out(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)",
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
    .op("aten::max_unpool3d.out(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toTensor(),
          (std::move(peek(*stack, 2, 6))).toIntVector(),
          (std::move(peek(*stack, 3, 6))).toIntVector(),
          (std::move(peek(*stack, 4, 6))).toIntVector());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, c10::optional<ScalarType>>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toTensor(),
          (std::move(peek(*stack, 1, 2))).toOptional<ScalarType>());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>>(unboxedKernel, (std::move(peek(*stack, 0, 4))).toTensor(),
          (std::move(peek(*stack, 1, 4))).toIntVector(),
          (std::move(peek(*stack, 2, 4))).toBool(),
          (std::move(peek(*stack, 3, 4))).toOptional<ScalarType>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::miopen_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
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
    .op("aten::miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
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
    .op("aten::miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor",
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
    .op("aten::miopen_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>, const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 14))).toTensor(),
          (std::move(peek(*stack, 1, 14))).toTensorVector(),
          (std::move(peek(*stack, 2, 14))).toInt(),
          (std::move(peek(*stack, 3, 14))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 4, 14)))),
          (std::move(peek(*stack, 5, 14))).toInt(),
          (std::move(peek(*stack, 6, 14))).toInt(),
          (std::move(peek(*stack, 7, 14))).toInt(),
          (std::move(peek(*stack, 8, 14))).toBool(),
          (std::move(peek(*stack, 9, 14))).toDouble(),
          (std::move(peek(*stack, 10, 14))).toBool(),
          (std::move(peek(*stack, 11, 14))).toBool(),
          (std::move(peek(*stack, 12, 14))).toIntVector(),
          toOptionalTensor((std::move(peek(*stack, 13, 14)))));
          drop(*stack, 14);
          pack(*stack, std::move(result_));
      })
    .op("aten::mkldnn_linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 3)))));
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::mse_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::multi_margin_loss.out(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 6, 7))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toScalar(),
          (std::move(peek(*stack, 3, 7))).toScalar(),
          toOptionalTensor((std::move(peek(*stack, 4, 7)))),
          (std::move(peek(*stack, 5, 7))).toInt());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::mv.out(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toInt());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>>(unboxedKernel, (std::move(peek(*stack, 0, 10))).toTensor(),
          (std::move(peek(*stack, 1, 10))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 10)))),
          toOptionalTensor((std::move(peek(*stack, 3, 10)))),
          toOptionalTensor((std::move(peek(*stack, 4, 10)))),
          toOptionalTensor((std::move(peek(*stack, 5, 10)))),
          toOptionalTensor((std::move(peek(*stack, 6, 10)))),
          (std::move(peek(*stack, 7, 10))).toBool(),
          (std::move(peek(*stack, 8, 10))).toDouble(),
          as_bool_array<3>((std::move(peek(*stack, 9, 10))).toBoolList()));
          drop(*stack, 10);
          pack(*stack, std::move(result_));
      })
    .op("aten::native_layer_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int M, int N, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, std::array<bool,3>>(unboxedKernel, (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          (std::move(peek(*stack, 2, 8))).toTensor(),
          (std::move(peek(*stack, 3, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 4, 8)))),
          (std::move(peek(*stack, 5, 8))).toInt(),
          (std::move(peek(*stack, 6, 8))).toInt(),
          as_bool_array<3>((std::move(peek(*stack, 7, 8))).toBoolList()));
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toScalar());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::new_empty(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          auto result_ = ((std::move(peek(*stack, 0, 6))).toTensor()).new_empty((std::move(peek(*stack, 1, 6))).toIntVector(),
          options);
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 5)))),
          (std::move(peek(*stack, 3, 5))).toInt(),
          (std::move(peek(*stack, 4, 5))).toInt());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 5)))),
          (std::move(peek(*stack, 3, 5))).toInt(),
          (std::move(peek(*stack, 4, 5))).toInt());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::nll_loss2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 7, 8))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          (std::move(peek(*stack, 2, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 8)))),
          (std::move(peek(*stack, 4, 8))).toInt(),
          (std::move(peek(*stack, 5, 8))).toInt(),
          (std::move(peek(*stack, 6, 8))).toTensor());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 5)))),
          (std::move(peek(*stack, 3, 5))).toInt(),
          (std::move(peek(*stack, 4, 5))).toInt());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 7, 8))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 8))).toTensor(),
          (std::move(peek(*stack, 1, 8))).toTensor(),
          (std::move(peek(*stack, 2, 8))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 3, 8)))),
          (std::move(peek(*stack, 4, 8))).toInt(),
          (std::move(peek(*stack, 5, 8))).toInt(),
          (std::move(peek(*stack, 6, 8))).toTensor());
          drop(*stack, 8);
          pack(*stack, std::move(result_));
      })
    .op("aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<std::tuple<Tensor, Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          toOptionalTensor((std::move(peek(*stack, 2, 5)))),
          (std::move(peek(*stack, 3, 5))).toInt(),
          (std::move(peek(*stack, 4, 5))).toInt());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, c10::optional<Scalar>, ScalarType>(unboxedKernel, (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toOptional<Scalar>(),
          (std::move(peek(*stack, 2, 3))).toScalarType());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toOptional<Scalar>(),
          (std::move(peek(*stack, 2, 5))).toIntVector(),
          (std::move(peek(*stack, 3, 5))).toBool(),
          (std::move(peek(*stack, 4, 5))).toScalarType());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)",
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
    .op("aten::ones.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toIntVector());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toInt());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t, bool, c10::optional<ScalarType>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toInt(),
          (std::move(peek(*stack, 2, 5))).toBool(),
          (std::move(peek(*stack, 3, 5))).toOptional<ScalarType>());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::promote_types(ScalarType type1, ScalarType type2) -> ScalarType",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<ScalarType, ScalarType, ScalarType>(unboxedKernel, (std::move(peek(*stack, 0, 2))).toScalarType(),
          (std::move(peek(*stack, 1, 2))).toScalarType());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points, int axis, ScalarType dtype) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          auto result_ = callUnboxedKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, ScalarType>(unboxedKernel, (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toTensor(),
          (std::move(peek(*stack, 2, 5))).toTensor(),
          (std::move(peek(*stack, 3, 5))).toInt(),
          (std::move(peek(*stack, 4, 5))).toScalarType());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::rad2deg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::randint.out(int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toInt(),
          (std::move(peek(*stack, 1, 3))).toIntVector());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::randint.generator_out(int high, int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, IntArrayRef, c10::optional<Generator>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toInt(),
          (std::move(peek(*stack, 1, 4))).toIntVector(),
          (std::move(peek(*stack, 2, 4))).toOptional<at::Generator>());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::randint.low_out(int low, int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 3, 4))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, int64_t, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 4))).toInt(),
          (std::move(peek(*stack, 1, 4))).toInt(),
          (std::move(peek(*stack, 2, 4))).toIntVector());
          drop(*stack, 4);
          pack(*stack, std::move(result_));
      })
    .op("aten::randint.low_generator_out(int low, int high, int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, int64_t, IntArrayRef, c10::optional<Generator>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toInt(),
          (std::move(peek(*stack, 1, 5))).toInt(),
          (std::move(peek(*stack, 2, 5))).toIntVector(),
          (std::move(peek(*stack, 3, 5))).toOptional<at::Generator>());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::randn(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 5))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::randn((std::move(peek(*stack, 0, 5))).toIntVector(),
          options);
          #else
              auto result_ = torch::randn((std::move(peek(*stack, 0, 5))).toIntVector(),
          options);
          #endif
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::randn.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::randn((std::move(peek(*stack, 0, 6))).toIntVector(),
          (std::move(peek(*stack, 1, 6))).toOptional<at::Generator>(),
          options);
          #else
              auto result_ = torch::randn((std::move(peek(*stack, 0, 6))).toIntVector(),
          (std::move(peek(*stack, 1, 6))).toOptional<at::Generator>(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::randn_like((std::move(peek(*stack, 0, 6))).toTensor(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #else
              auto result_ = torch::randn_like((std::move(peek(*stack, 0, 6))).toTensor(),
          options,
          (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::randperm.out(int n, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toInt());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::randperm.generator_out(int n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, c10::optional<Generator>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toInt(),
          (std::move(peek(*stack, 1, 3))).toOptional<at::Generator>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::range(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 2, 6))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 3, 6))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 4, 6))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 5, 6))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::range((std::move(peek(*stack, 0, 6))).toScalar(),
          (std::move(peek(*stack, 1, 6))).toScalar(),
          options);
          #else
              auto result_ = torch::range((std::move(peek(*stack, 0, 6))).toScalar(),
          (std::move(peek(*stack, 1, 6))).toScalar(),
          options);
          #endif
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::range.step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 3, 7))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 4, 7))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 5, 7))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 6, 7))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::range((std::move(peek(*stack, 0, 7))).toScalar(),
          (std::move(peek(*stack, 1, 7))).toScalar(),
          (std::move(peek(*stack, 2, 7))).toScalar(),
          options);
          #else
              auto result_ = torch::range((std::move(peek(*stack, 0, 7))).toScalar(),
          (std::move(peek(*stack, 1, 7))).toScalar(),
          (std::move(peek(*stack, 2, 7))).toScalar(),
          options);
          #endif
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::reflection_pad1d_backward.grad_input(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)",
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
    .op("aten::renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, Scalar, int64_t, Scalar>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toScalar(),
          (std::move(peek(*stack, 2, 5))).toInt(),
          (std::move(peek(*stack, 3, 5))).toScalar());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::replication_pad1d_backward.grad_input(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)",
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
    .op("aten::replication_pad3d.out(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toIntVector());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, IntArrayRef, c10::optional<MemoryFormat>>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toIntVector(),
          (std::move(peek(*stack, 2, 3))).toOptional<c10::MemoryFormat>());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor",
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
    .op("aten::rrelu_with_noise.out(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 6, 7))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toTensor(),
          (std::move(peek(*stack, 2, 7))).toScalar(),
          (std::move(peek(*stack, 3, 7))).toScalar(),
          (std::move(peek(*stack, 4, 7))).toBool(),
          (std::move(peek(*stack, 5, 7))).toOptional<at::Generator>());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
      
          const auto options = TensorOptions()
                  .dtype((std::move(peek(*stack, 1, 5))).toOptional<ScalarType>())
                  .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
                  .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
                  .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
          #ifdef USE_STATIC_DISPATCH
              auto result_ = at::scalar_tensor((std::move(peek(*stack, 0, 5))).toScalar(),
          options);
          #else
              auto result_ = torch::scalar_tensor((std::move(peek(*stack, 0, 5))).toScalar(),
          options);
          #endif
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)",
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
    .op("aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::sign_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::slow_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0) -> Tensor",
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
    .op("aten::slow_conv3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)",
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
    .op("aten::slow_conv_dilated3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1) -> Tensor",
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
    .op("aten::slow_conv_transpose2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::slow_conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1) -> Tensor",
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
    .op("aten::soft_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::sspaddmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toTensor());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::tanh_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, Scalar, Scalar>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toScalar(),
          (std::move(peek(*stack, 2, 3))).toScalar());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t, int64_t>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 3))).toInt(),
          (std::move(peek(*stack, 2, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, int64_t>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toInt());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::true_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 2, 3))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 3))).toTensor(),
          (std::move(peek(*stack, 1, 3))).toTensor());
          drop(*stack, 3);
          pack(*stack, std::move(result_));
      })
    .op("aten::trunc_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, int64_t>(unboxedKernel, self,
          (std::move(peek(*stack, 1, 2))).toInt());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_nearest1d_backward.grad_input(Tensor grad_output, int[1] output_size, int[3] input_size, float? scales=None, *, Tensor(a!) grad_input) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto grad_input = (std::move(peek(*stack, 4, 5))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>>(unboxedKernel, grad_input,
          (std::move(peek(*stack, 0, 5))).toTensor(),
          (std::move(peek(*stack, 1, 5))).toIntVector(),
          (std::move(peek(*stack, 2, 5))).toIntVector(),
          (std::move(peek(*stack, 3, 5))).toOptional<double>());
          drop(*stack, 5);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_nearest3d.out(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 5, 6))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 6))).toTensor(),
          (std::move(peek(*stack, 1, 6))).toIntVector(),
          (std::move(peek(*stack, 2, 6))).toOptional<double>(),
          (std::move(peek(*stack, 3, 6))).toOptional<double>(),
          (std::move(peek(*stack, 4, 6))).toOptional<double>());
          drop(*stack, 6);
          pack(*stack, std::move(result_));
      })
    .op("aten::upsample_trilinear3d.out(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 6, 7))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 7))).toTensor(),
          (std::move(peek(*stack, 1, 7))).toIntVector(),
          (std::move(peek(*stack, 2, 7))).toBool(),
          (std::move(peek(*stack, 3, 7))).toOptional<double>(),
          (std::move(peek(*stack, 4, 7))).toOptional<double>(),
          (std::move(peek(*stack, 5, 7))).toOptional<double>());
          drop(*stack, 7);
          pack(*stack, std::move(result_));
      })
    .op("aten::var.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)",
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
    .op("aten::zero_(Tensor(a!) self) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto self = (std::move(peek(*stack, 0, 1))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &>(unboxedKernel, self);
          drop(*stack, 1);
          pack(*stack, std::move(result_));
      })
    .op("aten::zeros.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)",
      [](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
          using namespace at;
          auto out = (std::move(peek(*stack, 1, 2))).toTensor();
          auto result_ = callUnboxedKernel<Tensor &, Tensor &, IntArrayRef>(unboxedKernel, out,
          (std::move(peek(*stack, 0, 2))).toIntVector());
          drop(*stack, 2);
          pack(*stack, std::move(result_));
      })
  ;

} // anon namespace


}} // namespace torch::jit
