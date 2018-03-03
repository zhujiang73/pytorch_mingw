#pragma once

// generated from tools/autograd/templates/Functions.h

#include <ATen/ATen.h>
#include <ATen/TensorGeometry.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"

namespace torch { namespace autograd { namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntList;
using at::Type;
using at::TensorGeometry;

inline std::vector<Tensor> unpack_list(at::ArrayRef<SavedVariable> xs) {
  // NB: we must explicitly do the conversion in the lambda, otherwise template
  // deduction will give a Tensor of Variable which is not convertible
  return fmap(xs, [](const SavedVariable& x) { return static_cast<Tensor>(x.unpack()); });
}

struct TypeAndSize {
  TypeAndSize() : type(nullptr) {}
  /* implicit */
  TypeAndSize(const Tensor & t)
    : sizes(t.sizes())
    , type(&t.type()) {}

  Tensor zeros() { return type->zeros(sizes); }

private:
  std::vector<int64_t> sizes;
  Type* type;
};

struct AbsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AbsBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct AcosBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AcosBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct AddBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AddBackward0"; }
  void release_variables() override {

  }
  


};
struct AddBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AddBackward1"; }
  void release_variables() override {

  }
  
  Scalar alpha;

};
struct AddbmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AddbmmBackward"; }
  void release_variables() override {
    batch2_.reset_data();
    batch1_.reset_data();
  }
  
  int64_t batch1_argsize_0;
  int64_t batch1_argsize_1;
  int64_t batch2_argsize_2;
  SavedVariable batch2_;
  Scalar alpha;
  SavedVariable batch1_;
  Scalar beta;

};
struct AddcdivBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AddcdivBackward"; }
  void release_variables() override {
    tensor2_.reset_data();
    tensor1_.reset_data();
  }
  
  SavedVariable tensor2_;
  Scalar value;
  SavedVariable tensor1_;

};
struct AddcmulBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AddcmulBackward"; }
  void release_variables() override {
    tensor2_.reset_data();
    tensor1_.reset_data();
  }
  
  SavedVariable tensor2_;
  Scalar value;
  SavedVariable tensor1_;

};
struct AddmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AddmmBackward"; }
  void release_variables() override {
    mat1_.reset_data();
    mat2_.reset_data();
  }
  
  std::vector<int64_t> mat1_sizes;
  SavedVariable mat1_;
  SavedVariable mat2_;
  Scalar alpha;
  std::vector<int64_t> mat2_sizes;
  Scalar beta;

};
struct AddmvBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AddmvBackward"; }
  void release_variables() override {
    vec_.reset_data();
    mat_.reset_data();
  }
  
  SavedVariable vec_;
  Scalar alpha;
  Scalar beta;
  SavedVariable mat_;

};
struct AddrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AddrBackward"; }
  void release_variables() override {
    vec2_.reset_data();
    vec1_.reset_data();
  }
  
  Scalar beta;
  SavedVariable vec2_;
  Scalar alpha;
  SavedVariable vec1_;

};
struct AliasBackward : public Function {
  using Function::Function;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AliasBackward"; }
  void release_variables() override {

  }
  


};
struct AsStridedBackward : public Function {
  using Function::Function;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AsStridedBackward"; }
  void release_variables() override {

  }
  
  TensorGeometry self_geometry;
  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  int64_t storage_offset;

};
struct AsinBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AsinBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct AtanBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AtanBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct Atan2Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "Atan2Backward"; }
  void release_variables() override {
    self_.reset_data();
    other_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable other_;

};
struct BaddbmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "BaddbmmBackward"; }
  void release_variables() override {
    batch2_.reset_data();
    batch1_.reset_data();
  }
  
  SavedVariable batch2_;
  Scalar alpha;
  SavedVariable batch1_;
  Scalar beta;

};
struct BernoulliBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "BernoulliBackward"; }
  void release_variables() override {

  }
  


};
struct BmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "BmmBackward"; }
  void release_variables() override {
    self_.reset_data();
    mat2_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable mat2_;

};
struct BtrifactBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "BtrifactBackward"; }
  void release_variables() override {

  }
  


};
struct BtrifactWithInfoBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "BtrifactWithInfoBackward"; }
  void release_variables() override {

  }
  


};
struct BtrisolveBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "BtrisolveBackward"; }
  void release_variables() override {

  }
  


};
struct CatBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CatBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> tensors_sizes_dim;
  int64_t dim;
  size_t tensors_size_;
};
struct CauchyBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CauchyBackward"; }
  void release_variables() override {

  }
  


};
struct CeilBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CeilBackward"; }
  void release_variables() override {

  }
  


};
struct ClampBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ClampBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar min;
  Scalar max;

};
struct ClampMinBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ClampMinBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar min;

};
struct ClampMaxBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ClampMaxBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar max;

};
struct CloneBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CloneBackward"; }
  void release_variables() override {

  }
  


};
struct CosBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CosBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct CoshBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CoshBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct CrossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CrossBackward"; }
  void release_variables() override {
    self_.reset_data();
    other_.reset_data();
  }
  
  SavedVariable self_;
  int64_t dim;
  SavedVariable other_;

};
struct CumprodBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CumprodBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  int64_t dim;

};
struct CumsumBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CumsumBackward"; }
  void release_variables() override {

  }
  
  int64_t dim;

};
struct ConvTbcBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ConvTbcBackward"; }
  void release_variables() override {
    self_.reset_data();
    weight_.reset_data();
    bias_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable weight_;
  SavedVariable bias_;
  int64_t pad;

};
struct DetWithSvdBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "DetWithSvdBackward"; }
  void release_variables() override {
    self_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable result0_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;

};
struct DiagBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "DiagBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;
  int64_t diagonal;

};
struct DistBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "DistBackward"; }
  void release_variables() override {
    self_.reset_data();
    other_.reset_data();
    result_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable other_;
  Scalar p;
  SavedVariable result_;

};
struct DivBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "DivBackward0"; }
  void release_variables() override {

  }
  
  Scalar other;

};
struct DivBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "DivBackward1"; }
  void release_variables() override {
    self_.reset_data();
    other_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable other_;

};
struct DotBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "DotBackward"; }
  void release_variables() override {
    tensor_.reset_data();
    self_.reset_data();
  }
  
  SavedVariable tensor_;
  SavedVariable self_;

};
struct EigBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "EigBackward"; }
  void release_variables() override {

  }
  


};
struct EqBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "EqBackward0"; }
  void release_variables() override {

  }
  
  TypeAndSize self_info;

};
struct EqBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "EqBackward1"; }
  void release_variables() override {

  }
  
  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct ErfBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ErfBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct ErfinvBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ErfinvBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct ExpBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ExpBackward"; }
  void release_variables() override {
    result_.reset_data();
  }
  
  SavedVariable result_;

};
struct Expm1Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "Expm1Backward"; }
  void release_variables() override {
    result_.reset_data();
  }
  
  SavedVariable result_;

};
struct ExpandBackward : public Function {
  using Function::Function;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ExpandBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;

};
struct ExponentialBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ExponentialBackward"; }
  void release_variables() override {

  }
  


};
struct FillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "FillBackward0"; }
  void release_variables() override {

  }
  


};
struct FillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "FillBackward1"; }
  void release_variables() override {

  }
  


};
struct FloorBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "FloorBackward"; }
  void release_variables() override {

  }
  


};
struct FmodBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "FmodBackward0"; }
  void release_variables() override {

  }
  


};
struct FmodBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "FmodBackward1"; }
  void release_variables() override {
    other_.reset_data();
  }
  
  SavedVariable other_;

};
struct FracBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "FracBackward"; }
  void release_variables() override {

  }
  


};
struct GatherBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "GatherBackward"; }
  void release_variables() override {
    index_.reset_data();
  }
  
  std::vector<int64_t> self_sizes;
  int64_t dim;
  SavedVariable index_;

};
struct GeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "GeBackward0"; }
  void release_variables() override {

  }
  
  TypeAndSize self_info;

};
struct GeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "GeBackward1"; }
  void release_variables() override {

  }
  
  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct GelsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "GelsBackward"; }
  void release_variables() override {

  }
  


};
struct GeometricBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "GeometricBackward"; }
  void release_variables() override {

  }
  


};
struct GeqrfBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "GeqrfBackward"; }
  void release_variables() override {

  }
  


};
struct GerBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "GerBackward"; }
  void release_variables() override {
    vec2_.reset_data();
    self_.reset_data();
  }
  
  SavedVariable vec2_;
  SavedVariable self_;

};
struct GesvBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "GesvBackward"; }
  void release_variables() override {
    A_.reset_data();
    solution_.reset_data();
  }
  
  SavedVariable A_;
  SavedVariable solution_;

};
struct GtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "GtBackward0"; }
  void release_variables() override {

  }
  
  TypeAndSize self_info;

};
struct GtBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "GtBackward1"; }
  void release_variables() override {

  }
  
  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct HistcBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "HistcBackward"; }
  void release_variables() override {

  }
  


};
struct IndexAddBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "IndexAddBackward"; }
  void release_variables() override {
    index_.reset_data();
  }
  
  int64_t dim;
  SavedVariable index_;

};
struct IndexCopyBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "IndexCopyBackward"; }
  void release_variables() override {
    index_.reset_data();
  }
  
  int64_t dim;
  SavedVariable index_;

};
struct IndexFillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "IndexFillBackward0"; }
  void release_variables() override {
    index_.reset_data();
  }
  
  int64_t dim;
  SavedVariable index_;

};
struct IndexFillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "IndexFillBackward1"; }
  void release_variables() override {
    index_.reset_data();
  }
  
  int64_t dim;
  SavedVariable index_;

};
struct IndexSelectBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "IndexSelectBackward"; }
  void release_variables() override {
    index_.reset_data();
  }
  
  std::vector<int64_t> self_sizes;
  int64_t dim;
  SavedVariable index_;

};
struct InverseBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "InverseBackward"; }
  void release_variables() override {
    output_.reset_data();
  }
  
  SavedVariable output_;

};
struct KthvalueBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "KthvalueBackward"; }
  void release_variables() override {
    indices_.reset_data();
  }
  
  std::vector<int64_t> self_sizes;
  int64_t dim;
  bool keepdim;
  SavedVariable indices_;

};
struct LeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LeBackward0"; }
  void release_variables() override {

  }
  
  TypeAndSize self_info;

};
struct LeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LeBackward1"; }
  void release_variables() override {

  }
  
  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct LerpBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LerpBackward"; }
  void release_variables() override {

  }
  
  Scalar weight;

};
struct LgammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LgammaBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct DigammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "DigammaBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct PolygammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "PolygammaBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  int64_t n;
  SavedVariable self_;

};
struct LogBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LogBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct Log1PBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "Log1PBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct LogNormalBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LogNormalBackward"; }
  void release_variables() override {

  }
  


};
struct LtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LtBackward0"; }
  void release_variables() override {

  }
  
  TypeAndSize self_info;

};
struct LtBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LtBackward1"; }
  void release_variables() override {

  }
  
  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct MaskedFillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MaskedFillBackward0"; }
  void release_variables() override {
    mask_.reset_data();
  }
  
  SavedVariable mask_;

};
struct MaskedFillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MaskedFillBackward1"; }
  void release_variables() override {
    mask_.reset_data();
  }
  
  SavedVariable mask_;

};
struct MaskedScatterBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MaskedScatterBackward"; }
  void release_variables() override {
    mask_.reset_data();
  }
  
  SavedVariable mask_;
  std::vector<int64_t> source_sizes;

};
struct MaskedSelectBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MaskedSelectBackward"; }
  void release_variables() override {
    mask_.reset_data();
  }
  
  TypeAndSize self_info;
  SavedVariable mask_;

};
struct MaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MaxBackward0"; }
  void release_variables() override {
    max_indices_.reset_data();
  }
  
  std::vector<int64_t> self_sizes;
  int64_t dim;
  bool keepdim;
  SavedVariable max_indices_;

};
struct MaxBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MaxBackward1"; }
  void release_variables() override {
    self_.reset_data();
    result_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable result_;

};
struct MaxBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MaxBackward2"; }
  void release_variables() override {
    self_.reset_data();
    other_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable other_;

};
struct MeanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MeanBackward0"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;
  int64_t dim;
  bool keepdim;

};
struct MeanBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MeanBackward1"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  std::vector<int64_t> self_sizes;
  SavedVariable self_;

};
struct MedianBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MedianBackward0"; }
  void release_variables() override {
    self_.reset_data();
    result_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable result_;

};
struct MedianBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MedianBackward1"; }
  void release_variables() override {
    indices_.reset_data();
  }
  
  std::vector<int64_t> self_sizes;
  int64_t dim;
  bool keepdim;
  SavedVariable indices_;

};
struct MinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MinBackward0"; }
  void release_variables() override {
    min_indices_.reset_data();
  }
  
  std::vector<int64_t> self_sizes;
  int64_t dim;
  bool keepdim;
  SavedVariable min_indices_;

};
struct MinBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MinBackward1"; }
  void release_variables() override {
    self_.reset_data();
    result_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable result_;

};
struct MinBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MinBackward2"; }
  void release_variables() override {
    self_.reset_data();
    other_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable other_;

};
struct MmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MmBackward"; }
  void release_variables() override {
    self_.reset_data();
    mat2_.reset_data();
  }
  
  SavedVariable self_;
  std::vector<int64_t> mat2_sizes;
  SavedVariable mat2_;
  std::vector<int64_t> self_sizes;

};
struct ModeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ModeBackward"; }
  void release_variables() override {
    indices_.reset_data();
  }
  
  std::vector<int64_t> self_sizes;
  int64_t dim;
  bool keepdim;
  SavedVariable indices_;

};
struct MulBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MulBackward0"; }
  void release_variables() override {

  }
  
  Scalar other;

};
struct MulBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MulBackward1"; }
  void release_variables() override {
    self_.reset_data();
    other_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable other_;

};
struct MvBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MvBackward"; }
  void release_variables() override {
    vec_.reset_data();
    self_.reset_data();
  }
  
  SavedVariable vec_;
  SavedVariable self_;

};
struct NeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NeBackward0"; }
  void release_variables() override {

  }
  
  TypeAndSize self_info;

};
struct NeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NeBackward1"; }
  void release_variables() override {

  }
  
  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct NegBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NegBackward"; }
  void release_variables() override {

  }
  


};
struct NormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NormBackward0"; }
  void release_variables() override {
    self_.reset_data();
    result_.reset_data();
  }
  
  SavedVariable self_;
  Scalar p;
  SavedVariable result_;

};
struct NormBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NormBackward1"; }
  void release_variables() override {
    self_.reset_data();
    result_.reset_data();
  }
  
  SavedVariable self_;
  Scalar p;
  int64_t dim;
  bool keepdim;
  SavedVariable result_;

};
struct NormalBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NormalBackward0"; }
  void release_variables() override {

  }
  


};
struct NormalBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NormalBackward1"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> mean_sizes;

};
struct NormalBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NormalBackward2"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> std_sizes;

};
struct NormalBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NormalBackward3"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> mean_sizes;
  std::vector<int64_t> std_sizes;

};
struct OrgqrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "OrgqrBackward"; }
  void release_variables() override {

  }
  


};
struct OrmqrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "OrmqrBackward"; }
  void release_variables() override {

  }
  


};
struct PermuteBackward : public Function {
  using Function::Function;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "PermuteBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> dims;

};
struct PoissonBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "PoissonBackward"; }
  void release_variables() override {

  }
  
  TypeAndSize self_info;

};
struct PotrfBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "PotrfBackward"; }
  void release_variables() override {
    output_.reset_data();
  }
  
  bool upper;
  SavedVariable output_;

};
struct PotriBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "PotriBackward"; }
  void release_variables() override {

  }
  


};
struct PotrsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "PotrsBackward"; }
  void release_variables() override {

  }
  


};
struct PowBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "PowBackward0"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar exponent;

};
struct PowBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "PowBackward1"; }
  void release_variables() override {
    self_.reset_data();
    exponent_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable exponent_;

};
struct ProdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ProdBackward0"; }
  void release_variables() override {
    self_.reset_data();
    result_.reset_data();
  }
  
  SavedVariable self_;
  int64_t dim;
  bool keepdim;
  SavedVariable result_;

};
struct ProdBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ProdBackward1"; }
  void release_variables() override {
    self_.reset_data();
    result_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable result_;

};
struct PstrfBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "PstrfBackward"; }
  void release_variables() override {

  }
  


};
struct PutBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "PutBackward"; }
  void release_variables() override {
    index_.reset_data();
  }
  
  SavedVariable index_;
  TypeAndSize source_info;
  bool accumulate;

};
struct QrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "QrBackward"; }
  void release_variables() override {

  }
  


};
struct RandomBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "RandomBackward0"; }
  void release_variables() override {

  }
  


};
struct RandomBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "RandomBackward1"; }
  void release_variables() override {

  }
  


};
struct RandomBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "RandomBackward2"; }
  void release_variables() override {

  }
  


};
struct ReciprocalBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ReciprocalBackward"; }
  void release_variables() override {
    result_.reset_data();
  }
  
  SavedVariable result_;

};
struct RemainderBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "RemainderBackward0"; }
  void release_variables() override {

  }
  


};
struct RemainderBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "RemainderBackward1"; }
  void release_variables() override {

  }
  


};
struct RenormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "RenormBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar p;
  int64_t dim;
  Scalar maxnorm;

};
struct RepeatBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "RepeatBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  std::vector<int64_t> repeats;

};
struct Roipooling2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "Roipooling2DBackward"; }
  void release_variables() override {
    input_.reset_data();
    rois_.reset_data();
    result1_.reset_data();
  }
  
  SavedVariable input_;
  SavedVariable rois_;
  int64_t pooledHeight;
  int64_t pooledWidth;
  double spatialScale;
  SavedVariable result1_;

};
struct RoundBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "RoundBackward"; }
  void release_variables() override {

  }
  


};
struct RsqrtBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "RsqrtBackward"; }
  void release_variables() override {
    result_.reset_data();
  }
  
  SavedVariable result_;

};
struct ScatterBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ScatterBackward0"; }
  void release_variables() override {
    index_.reset_data();
  }
  
  int64_t dim;
  SavedVariable index_;

};
struct ScatterBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ScatterBackward1"; }
  void release_variables() override {
    index_.reset_data();
  }
  
  int64_t dim;
  SavedVariable index_;

};
struct ScatterAddBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ScatterAddBackward"; }
  void release_variables() override {
    index_.reset_data();
  }
  
  int64_t dim;
  SavedVariable index_;

};
struct SigmoidBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SigmoidBackward"; }
  void release_variables() override {
    result_.reset_data();
  }
  
  SavedVariable result_;

};
struct SignBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SignBackward"; }
  void release_variables() override {

  }
  


};
struct SinBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SinBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct SinhBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SinhBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct SortBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SortBackward"; }
  void release_variables() override {
    indices_.reset_data();
  }
  
  std::vector<int64_t> self_sizes;
  int64_t dim;
  SavedVariable indices_;

};
struct SplitBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SplitBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  std::vector<int64_t> self_sizes;
  SavedVariable self_;
  int64_t split_size;
  int64_t dim;

};
struct SqrtBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SqrtBackward"; }
  void release_variables() override {
    result_.reset_data();
  }
  
  SavedVariable result_;

};
struct SqueezeBackward0 : public Function {
  using Function::Function;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SqueezeBackward0"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;

};
struct SqueezeBackward1 : public Function {
  using Function::Function;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SqueezeBackward1"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;
  int64_t dim;

};
struct StdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "StdBackward0"; }
  void release_variables() override {
    self_.reset_data();
    result_.reset_data();
  }
  
  SavedVariable self_;
  bool unbiased;
  SavedVariable result_;

};
struct StdBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "StdBackward1"; }
  void release_variables() override {
    self_.reset_data();
    result_.reset_data();
  }
  
  SavedVariable self_;
  int64_t dim;
  bool unbiased;
  bool keepdim;
  SavedVariable result_;

};
struct SubBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SubBackward0"; }
  void release_variables() override {

  }
  


};
struct SubBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SubBackward1"; }
  void release_variables() override {

  }
  
  Scalar alpha;

};
struct SumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SumBackward0"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;

};
struct SumBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SumBackward1"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;
  int64_t dim;
  bool keepdim;

};
struct SvdBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SvdBackward"; }
  void release_variables() override {
    self_.reset_data();
    res1_.reset_data();
    res2_.reset_data();
    res3_.reset_data();
  }
  
  SavedVariable self_;
  bool some;
  SavedVariable res1_;
  SavedVariable res2_;
  SavedVariable res3_;

};
struct SymeigBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SymeigBackward"; }
  void release_variables() override {

  }
  


};
struct TBackward : public Function {
  using Function::Function;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "TBackward"; }
  void release_variables() override {

  }
  


};
struct TakeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "TakeBackward"; }
  void release_variables() override {
    index_.reset_data();
  }
  
  TypeAndSize self_info;
  SavedVariable index_;

};
struct TanBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "TanBackward"; }
  void release_variables() override {
    result_.reset_data();
  }
  
  SavedVariable result_;

};
struct TanhBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "TanhBackward"; }
  void release_variables() override {
    result_.reset_data();
  }
  
  SavedVariable result_;

};
struct TopkBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "TopkBackward"; }
  void release_variables() override {
    indices_.reset_data();
  }
  
  std::vector<int64_t> self_sizes;
  int64_t dim;
  SavedVariable indices_;

};
struct TraceBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "TraceBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;

};
struct TransposeBackward : public Function {
  using Function::Function;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "TransposeBackward"; }
  void release_variables() override {

  }
  
  int64_t dim0;
  int64_t dim1;

};
struct TrilBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "TrilBackward"; }
  void release_variables() override {

  }
  
  int64_t diagonal;

};
struct TriuBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "TriuBackward"; }
  void release_variables() override {

  }
  
  int64_t diagonal;

};
struct TrtrsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "TrtrsBackward"; }
  void release_variables() override {
    self_.reset_data();
    A_.reset_data();
    res1_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable A_;
  bool upper;
  bool transpose;
  bool unitriangular;
  SavedVariable res1_;

};
struct TruncBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "TruncBackward"; }
  void release_variables() override {

  }
  


};
struct UnfoldBackward : public Function {
  using Function::Function;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UnfoldBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;
  int64_t dimension;
  int64_t size;
  int64_t step;

};
struct UniformBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UniformBackward"; }
  void release_variables() override {

  }
  


};
struct UnsafeViewBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UnsafeViewBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;

};
struct UnsqueezeBackward : public Function {
  using Function::Function;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UnsqueezeBackward"; }
  void release_variables() override {

  }
  
  int64_t dim;

};
struct VarBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "VarBackward0"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  bool unbiased;

};
struct VarBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "VarBackward1"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  int64_t dim;
  bool unbiased;
  bool keepdim;

};
struct ViewBackward : public Function {
  using Function::Function;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ViewBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;

};
struct SWhereBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SWhereBackward"; }
  void release_variables() override {
    condition_.reset_data();
  }
  
  SavedVariable condition_;

};
struct ZeroBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ZeroBackward"; }
  void release_variables() override {

  }
  


};
struct SparseMaskBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SparseMaskBackward"; }
  void release_variables() override {

  }
  


};
struct StandardGammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "StandardGammaBackward"; }
  void release_variables() override {
    self_.reset_data();
    output_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable output_;

};
struct StandardGammaGradBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "StandardGammaGradBackward"; }
  void release_variables() override {

  }
  


};
struct BinaryCrossEntropyBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "BinaryCrossEntropyBackward"; }
  void release_variables() override {
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  bool size_average;
  bool reduce;

};
struct EmbeddingBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "EmbeddingBackward"; }
  void release_variables() override {
    indices_.reset_data();
  }
  
  int64_t weight_argsize_0;
  SavedVariable indices_;
  int64_t padding_idx;
  bool scale_grad_by_freq;
  bool sparse;

};
struct EmbeddingBagBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "EmbeddingBagBackward"; }
  void release_variables() override {
    indices_.reset_data();
    offsets_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }
  
  int64_t weight_argsize_0;
  SavedVariable indices_;
  SavedVariable offsets_;
  bool scale_grad_by_freq;
  int64_t mode;
  bool sparse;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct EmbeddingRenormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "EmbeddingRenormBackward"; }
  void release_variables() override {

  }
  


};
struct KlDivBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "KlDivBackward"; }
  void release_variables() override {
    self_.reset_data();
    target_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable target_;
  bool size_average;
  bool reduce;

};
struct L1LossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "L1LossBackward"; }
  void release_variables() override {
    self_.reset_data();
    target_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable target_;
  bool size_average;
  bool reduce;

};
struct MseLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MseLossBackward"; }
  void release_variables() override {
    self_.reset_data();
    target_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable target_;
  bool size_average;
  bool reduce;

};
struct MultiMarginLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MultiMarginLossBackward"; }
  void release_variables() override {
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable target_;
  Scalar p;
  Scalar margin;
  SavedVariable weight_;
  bool size_average;

};
struct MultilabelMarginLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MultilabelMarginLossBackward"; }
  void release_variables() override {
    self_.reset_data();
    target_.reset_data();
    is_target_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable target_;
  bool size_average;
  bool reduce;
  SavedVariable is_target_;

};
struct NllLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NllLossBackward"; }
  void release_variables() override {
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
    total_weight_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  bool size_average;
  int64_t ignore_index;
  bool reduce;
  SavedVariable total_weight_;

};
struct NllLoss2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NllLoss2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
    total_weight_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  bool size_average;
  int64_t ignore_index;
  bool reduce;
  SavedVariable total_weight_;

};
struct SmoothL1LossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SmoothL1LossBackward"; }
  void release_variables() override {
    self_.reset_data();
    target_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable target_;
  bool size_average;
  bool reduce;

};
struct SoftMarginLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SoftMarginLossBackward"; }
  void release_variables() override {
    self_.reset_data();
    target_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable target_;
  bool size_average;
  bool reduce;

};
struct EluBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "EluBackward"; }
  void release_variables() override {
    output_.reset_data();
  }
  
  Scalar alpha;
  Scalar scale;
  SavedVariable output_;

};
struct GluBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "GluBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  int64_t dim;

};
struct HardshrinkBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "HardshrinkBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar lambd;

};
struct HardtanhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "HardtanhBackward0"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar min_val;
  Scalar max_val;

};
struct HardtanhBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "HardtanhBackward1"; }
  void release_variables() override {
    output_.reset_data();
  }
  
  Scalar min_val;
  Scalar max_val;
  SavedVariable output_;

};
struct LeakyReluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LeakyReluBackward0"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar negative_slope;

};
struct LeakyReluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LeakyReluBackward1"; }
  void release_variables() override {
    output_.reset_data();
  }
  
  Scalar negative_slope;
  SavedVariable output_;

};
struct LogSigmoidBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LogSigmoidBackward"; }
  void release_variables() override {
    self_.reset_data();
    buffer_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable buffer_;

};
struct LogSoftmaxBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LogSoftmaxBackward"; }
  void release_variables() override {
    self_.reset_data();
    output_.reset_data();
  }
  
  SavedVariable self_;
  int64_t dim;
  SavedVariable output_;

};
struct PreluBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "PreluBackward"; }
  void release_variables() override {
    self_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable weight_;

};
struct RreluWithNoiseBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "RreluWithNoiseBackward0"; }
  void release_variables() override {
    self_.reset_data();
    noise_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable noise_;
  Scalar lower;
  Scalar upper;
  bool training;

};
struct RreluWithNoiseBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "RreluWithNoiseBackward1"; }
  void release_variables() override {
    noise_.reset_data();
    output_.reset_data();
  }
  
  SavedVariable noise_;
  Scalar lower;
  Scalar upper;
  bool training;
  SavedVariable output_;

};
struct SoftmaxBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SoftmaxBackward"; }
  void release_variables() override {
    self_.reset_data();
    output_.reset_data();
  }
  
  SavedVariable self_;
  int64_t dim;
  SavedVariable output_;

};
struct SoftplusBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SoftplusBackward"; }
  void release_variables() override {
    self_.reset_data();
    output_.reset_data();
  }
  
  SavedVariable self_;
  Scalar beta;
  Scalar threshold;
  SavedVariable output_;

};
struct SoftshrinkBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SoftshrinkBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar lambd;

};
struct ThresholdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThresholdBackward0"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar threshold;
  Scalar value;

};
struct ThresholdBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThresholdBackward1"; }
  void release_variables() override {
    output_.reset_data();
  }
  
  Scalar threshold;
  Scalar value;
  SavedVariable output_;

};
struct ReflectionPad1DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ReflectionPad1DBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct ReflectionPad2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ReflectionPad2DBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct ReplicationPad1DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ReplicationPad1DBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct ReplicationPad2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ReplicationPad2DBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct ReplicationPad3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ReplicationPad3DBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct UpsampleLinear1DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UpsampleLinear1DBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;

};
struct UpsampleBilinear2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UpsampleBilinear2DBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;

};
struct UpsampleTrilinear3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UpsampleTrilinear3DBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;

};
struct UpsampleNearest1DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UpsampleNearest1DBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  int64_t scale_factor;

};
struct UpsampleNearest2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UpsampleNearest2DBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  int64_t scale_factor;

};
struct UpsampleNearest3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UpsampleNearest3DBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  int64_t scale_factor;

};
struct AdaptiveAvgPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AdaptiveAvgPool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct AdaptiveAvgPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AdaptiveAvgPool3DBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;

};
struct AdaptiveMaxPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AdaptiveMaxPool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    indices_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable indices_;

};
struct AdaptiveMaxPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AdaptiveMaxPool3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    indices_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable indices_;

};
struct AvgPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AvgPool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;

};
struct AvgPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AvgPool3DBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;

};
struct FractionalMaxPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "FractionalMaxPool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    indices_.reset_data();
  }
  
  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> output_size;
  SavedVariable indices_;

};
struct MaxPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MaxPool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    indices_.reset_data();
  }
  
  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable indices_;

};
struct MaxPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MaxPool3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    indices_.reset_data();
  }
  
  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable indices_;

};
struct MaxUnpool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MaxUnpool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    indices_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable indices_;
  std::vector<int64_t> output_size;

};
struct MaxUnpool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MaxUnpool3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    indices_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable indices_;
  std::vector<int64_t> output_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct ThnnBatchNormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnBatchNormBackward"; }
  void release_variables() override {
    self_.reset_data();
    weight_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    save_mean_.reset_data();
    save_std_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  double eps;
  SavedVariable save_mean_;
  SavedVariable save_std_;

};
struct ThnnBatchNormBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnBatchNormBackwardBackward"; }
  void release_variables() override {
    save_mean_.reset_data();
    save_std_.reset_data();
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
  }
  
  SavedVariable save_mean_;
  SavedVariable save_std_;
  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  double eps;

};
struct ThnnConvTranspose2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConvTranspose2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    weight_.reset_data();
    columns_.reset_data();
    ones_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;
  SavedVariable columns_;
  SavedVariable ones_;

};
struct ThnnConvTranspose2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConvTranspose2DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct ThnnConvTranspose3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConvTranspose3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    weight_.reset_data();
    finput_.reset_data();
    fgrad_input_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;
  SavedVariable finput_;
  SavedVariable fgrad_input_;

};
struct ThnnConvTranspose3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConvTranspose3DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct ThnnConv2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConv2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    weight_.reset_data();
    finput_.reset_data();
    fgrad_input_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  SavedVariable finput_;
  SavedVariable fgrad_input_;

};
struct ThnnConv2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConv2DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct ThnnConvDepthwise2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConvDepthwise2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct ThnnConvDepthwise2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConvDepthwise2DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable grad_output_;
  int64_t self_argsize_1;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct ThnnConv3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConv3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    weight_.reset_data();
    finput_.reset_data();
    fgrad_input_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  SavedVariable finput_;
  SavedVariable fgrad_input_;

};
struct ThnnConv3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConv3DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct ThnnConvDilated2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConvDilated2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    weight_.reset_data();
    columns_.reset_data();
    ones_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  SavedVariable columns_;
  SavedVariable ones_;

};
struct ThnnConvDilated2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConvDilated2DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct ThnnConvDilated3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConvDilated3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    weight_.reset_data();
    columns_.reset_data();
    ones_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  SavedVariable columns_;
  SavedVariable ones_;

};
struct ThnnConvDilated3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThnnConvDilated3DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct AdaptiveAvgPool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AdaptiveAvgPool2DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
  }
  
  SavedVariable grad_output_;
  TypeAndSize self_info;

};
struct AdaptiveAvgPool3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AdaptiveAvgPool3DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
  }
  
  SavedVariable grad_output_;
  TypeAndSize self_info;

};
struct AdaptiveMaxPool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AdaptiveMaxPool2DBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    indices_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable indices_;
  TypeAndSize self_info;

};
struct AdaptiveMaxPool3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AdaptiveMaxPool3DBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    indices_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable indices_;
  TypeAndSize self_info;

};
struct AvgPool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AvgPool2DBackwardBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;
  TypeAndSize self_info;

};
struct AvgPool3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "AvgPool3DBackwardBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;
  TypeAndSize self_info;

};
struct EluBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "EluBackwardBackward"; }
  void release_variables() override {
    output_.reset_data();
    grad_output_.reset_data();
  }
  
  Scalar alpha;
  Scalar scale;
  SavedVariable output_;
  SavedVariable grad_output_;

};
struct FractionalMaxPool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "FractionalMaxPool2DBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    indices_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable indices_;
  TypeAndSize self_info;

};
struct GluBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "GluBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    grad_output_.reset_data();
  }
  
  SavedVariable self_;
  int64_t dim;
  SavedVariable grad_output_;

};
struct HardshrinkBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "HardshrinkBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar lambd;

};
struct HardtanhBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "HardtanhBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar min_val;
  Scalar max_val;

};
struct KlDivBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "KlDivBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    target_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable target_;
  bool size_average;
  bool reduce;

};
struct L1LossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "L1LossBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    target_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable target_;
  bool size_average;
  bool reduce;

};
struct LogSigmoidBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LogSigmoidBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    buffer_.reset_data();
    grad_output_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable buffer_;
  SavedVariable grad_output_;

};
struct LogSoftmaxBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LogSoftmaxBackwardBackward"; }
  void release_variables() override {
    output_.reset_data();
    grad_output_.reset_data();
  }
  
  int64_t dim;
  SavedVariable output_;
  SavedVariable grad_output_;

};
struct LeakyReluBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "LeakyReluBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar negative_slope;

};
struct MaxPool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MaxPool2DBackwardBackward"; }
  void release_variables() override {
    indices_.reset_data();
  }
  
  SavedVariable indices_;
  TypeAndSize self_info;

};
struct MaxUnpool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MaxUnpool2DBackwardBackward"; }
  void release_variables() override {
    indices_.reset_data();
  }
  
  SavedVariable indices_;
  std::vector<int64_t> output_size;
  TypeAndSize self_info;

};
struct MseLossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "MseLossBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }
  
  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  bool size_average;
  bool reduce;

};
struct NllLossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NllLossBackwardBackward"; }
  void release_variables() override {
    target_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable target_;
  SavedVariable weight_;
  bool size_average;
  int64_t ignore_index;
  bool reduce;

};
struct NllLoss2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NllLoss2DBackwardBackward"; }
  void release_variables() override {
    target_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable target_;
  SavedVariable weight_;
  bool size_average;
  int64_t ignore_index;
  bool reduce;

};
struct PreluBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "PreluBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;

};
struct RreluWithNoiseBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "RreluWithNoiseBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    noise_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable noise_;
  Scalar lower;
  Scalar upper;
  bool training;

};
struct ReflectionPad1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ReflectionPad1DBackwardBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct ReflectionPad2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ReflectionPad2DBackwardBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct ReplicationPad1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ReplicationPad1DBackwardBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct ReplicationPad2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ReplicationPad2DBackwardBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct ReplicationPad3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ReplicationPad3DBackwardBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct SmoothL1LossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SmoothL1LossBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }
  
  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  bool size_average;
  bool reduce;

};
struct SoftplusBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SoftplusBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    output_.reset_data();
    grad_output_.reset_data();
  }
  
  SavedVariable self_;
  Scalar beta;
  Scalar threshold;
  SavedVariable output_;
  SavedVariable grad_output_;

};
struct SoftmaxBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SoftmaxBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    output_.reset_data();
    grad_output_.reset_data();
  }
  
  SavedVariable self_;
  int64_t dim;
  SavedVariable output_;
  SavedVariable grad_output_;

};
struct SoftMarginLossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SoftMarginLossBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }
  
  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  bool size_average;
  bool reduce;

};
struct SoftshrinkBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SoftshrinkBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar lambd;

};
struct ThresholdBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "ThresholdBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
  }
  
  SavedVariable self_;
  Scalar threshold;
  Scalar value;

};
struct UpsampleLinear1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UpsampleLinear1DBackwardBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> output_size;

};
struct UpsampleBilinear2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UpsampleBilinear2DBackwardBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> output_size;

};
struct UpsampleTrilinear3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UpsampleTrilinear3DBackwardBackward"; }
  void release_variables() override {

  }
  
  std::vector<int64_t> output_size;

};
struct UpsampleNearest1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UpsampleNearest1DBackwardBackward"; }
  void release_variables() override {

  }
  
  int64_t scale_factor;

};
struct UpsampleNearest2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UpsampleNearest2DBackwardBackward"; }
  void release_variables() override {

  }
  
  int64_t scale_factor;

};
struct UpsampleNearest3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "UpsampleNearest3DBackwardBackward"; }
  void release_variables() override {

  }
  
  int64_t scale_factor;

};
struct SigmoidBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "SigmoidBackwardBackward"; }
  void release_variables() override {
    output_.reset_data();
    grad_output_.reset_data();
  }
  
  SavedVariable output_;
  SavedVariable grad_output_;

};
struct TanhBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "TanhBackwardBackward"; }
  void release_variables() override {
    output_.reset_data();
    grad_output_.reset_data();
  }
  
  SavedVariable output_;
  SavedVariable grad_output_;

};
struct CudnnConvolutionTransposeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CudnnConvolutionTransposeBackward"; }
  void release_variables() override {
    self_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups;
  bool benchmark;
  bool deterministic;

};
struct CudnnConvolutionTransposeBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CudnnConvolutionTransposeBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    grad_output_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups;
  bool benchmark;
  bool deterministic;

};
struct CudnnConvolutionBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CudnnConvolutionBackward"; }
  void release_variables() override {
    self_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups;
  bool benchmark;
  bool deterministic;

};
struct CudnnConvolutionBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CudnnConvolutionBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    grad_output_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups;
  bool benchmark;
  bool deterministic;

};
struct CudnnGridSamplerBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CudnnGridSamplerBackward"; }
  void release_variables() override {
    self_.reset_data();
    grid_.reset_data();
  }
  
  SavedVariable self_;
  SavedVariable grid_;

};
struct CudnnAffineGridGeneratorBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CudnnAffineGridGeneratorBackward"; }
  void release_variables() override {

  }
  
  int64_t N;
  int64_t C;
  int64_t H;
  int64_t W;

};
struct CudnnBatchNormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CudnnBatchNormBackward"; }
  void release_variables() override {
    input_.reset_data();
    weight_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }
  
  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  double epsilon;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct CudnnBatchNormBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CudnnBatchNormBackwardBackward"; }
  void release_variables() override {
    input_.reset_data();
    grad_output_.reset_data();
    weight_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    save_mean_.reset_data();
    save_var_.reset_data();
  }
  
  SavedVariable input_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_var_;
  double epsilon;

};
struct NnpackSpatialConvolutionBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "NnpackSpatialConvolutionBackward"; }
  void release_variables() override {
    input_.reset_data();
    weight_.reset_data();
  }
  
  SavedVariable input_;
  SavedVariable weight_;
  int64_t kW;
  int64_t kH;
  int64_t padW;
  int64_t padH;
  std::vector<int64_t> weight_sizes;

};
struct CudnnRnnBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "CudnnRnnBackward"; }
  void release_variables() override {
    input_.reset_data();
    weight_.clear();
    hx_.reset_data();
    cx_.reset_data();
    dropout_state_.reset_data();
    result0_.reset_data();
    result3_.reset_data();
    result4_.reset_data();
  }
  bool retain_variables = true;
virtual void will_release_variables() override {
  retain_variables = false;
}

  SavedVariable input_;
  std::vector<SavedVariable> weight_;
  int64_t weight_stride0;
  SavedVariable hx_;
  SavedVariable cx_;
  int64_t mode;
  int64_t hidden_size;
  int64_t num_layers;
  bool batch_first;
  double dropout;
  bool train;
  bool bidirectional;
  std::vector<int64_t> batch_sizes;
  SavedVariable dropout_state_;
  SavedVariable result0_;
  SavedVariable result3_;
  SavedVariable result4_;
  size_t weight_size_;
};

}}} // namespace torch::autograd::generated
