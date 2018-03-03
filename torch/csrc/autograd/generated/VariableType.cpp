#include "Python.h"
#include "VariableType.h"

// generated from tools/autograd/templates/VariableType.cpp

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/utils/variadic.h"

#include <initializer_list>
#include <iostream>
#include <functional>
#include <cstddef>

#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd {
// Helper methods for working with Attributes (torch/csrc/jit/attributes.h)

// The overloaded accessors are convenient for the generated code (since we
// don't want to make the codegen do the dispatch manually)
static void setattr(jit::Node* n, jit::Symbol name, int64_t v)             { n->i_(name, v); }
static void setattr(jit::Node* n, jit::Symbol name, const at::Scalar& v)   { n->t_(name, v.toTensor()); }
static void setattr(jit::Node* n, jit::Symbol name, SparseTensor s)        { n->t_(name, s.tref); }
static void setattr(jit::Node* n, jit::Symbol name, const at::IntList& v)  { n->is_(name, v); }
static void setattr(jit::Node* n, jit::Symbol name, bool v)                { n->i_(name, v); }
static void setattr(jit::Node* n, jit::Symbol name, double v)              { n->f_(name, v); }
template<std::size_t N>
static void setattr(jit::Node* n, jit::Symbol name, std::array<bool, N> v) { n->is_(name, std::vector<int64_t>(v.begin(), v.end())); }

VariableType::VariableType(Context* context, Type* baseType)
  : Type(context)
  , baseType(baseType) {
  str = std::string("Variable[") + baseType->toString() + "]";
}

ScalarType VariableType::scalarType() const {
  return baseType->scalarType();
}
Backend VariableType::backend() const {
  return baseType->backend();
}
bool VariableType::is_cuda() const { return baseType->is_cuda(); }
bool VariableType::is_sparse() const { return baseType->is_sparse(); }
bool VariableType::is_distributed() const { return baseType->is_distributed(); }

std::unique_ptr<Storage> VariableType::storage() const {
  return baseType->storage();
}
std::unique_ptr<Storage> VariableType::storage(size_t size) const {
  return baseType->storage(size);
}
std::unique_ptr<Storage> VariableType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
  return baseType->storageFromBlob(data, size, deleter);
}
std::unique_ptr<Storage> VariableType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  return baseType->unsafeStorageFromTH(th_pointer, retain);
}
std::unique_ptr<Storage> VariableType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
  return baseType->storageWithAllocator(size, std::move(allocator));
}
Tensor VariableType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  return make_variable(baseType->unsafeTensorFromTH(th_pointer, retain), /*requires_grad=*/false);
}
std::unique_ptr<Generator> VariableType::generator() const {
  return baseType->generator();
}

const char * VariableType::toString() const {
  return str.c_str();
}
size_t VariableType::elementSizeInBytes() const {
  return baseType->elementSizeInBytes();
}
Type & VariableType::toBackend(Backend b) const {
  return *getType(baseType->toBackend(b));
}
Type & VariableType::toScalarType(ScalarType s) const {
  return *getType(baseType->toScalarType(s));
}
TypeID VariableType::ID() const {
  throw std::runtime_error("VariableType::ID() not implemented");
}

const char * VariableType::typeString() {
  return "VariableType";
}

struct VariableTypeRegistry {
  static constexpr int MaxTypes = static_cast<int>(at::TypeID::NumOptions);

  VariableTypeRegistry();

  std::vector<VariableType> types_vec;
  at::Type* types[MaxTypes];
};

VariableTypeRegistry::VariableTypeRegistry() {
  auto& context = at::globalContext();
  types_vec.reserve(MaxTypes);
  memset(types, 0, MaxTypes * sizeof(at::Type*));
  for (int p = 0; p < static_cast<int>(Backend::NumOptions); ++p) {
    for (int s = 0; s < static_cast<int>(ScalarType::NumOptions); s++) {
      auto baseType = context.type_registry[p][s].get();
      if (baseType && baseType->backend() != Backend::Undefined) {
        auto id = static_cast<int>(baseType->ID());
        types_vec.emplace_back(&context, baseType);
        types[id] = &types_vec.back();
      }
    }
  }
}

static VariableTypeRegistry registry;

bool VariableType::isVariableType(const at::Type& type) {
  // Since all VariableTypes are allocated contiguously in types_vec, we can
  // just check that the pointer is inside the correct range.
  ptrdiff_t offset = (char*)&type - (char*)registry.types_vec.data();
  ptrdiff_t extent = VariableTypeRegistry::MaxTypes * sizeof(VariableType);
  return offset >= 0 && offset < extent;
}

at::Type* VariableType::getType(const at::Type& baseType) {
  return registry.types[static_cast<int>(baseType.ID())];
}

at::Type* VariableType::getType(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    throw std::runtime_error("tensor is undefined");
  }
  return getType(tensor.type());
}

std::vector<at::Type*> VariableType::allTypes() {
  std::vector<Type*> res;
  res.reserve(registry.types_vec.size());
  for (auto& type : registry.types_vec) {
    res.push_back(&type);
  }
  return res;
}

Variable & VariableType::checked_cast_variable(const Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    runtime_error("Expected a Tensor of type Variable but found an undefined Tensor for argument #%d '%s'",
        pos, name);
  }
  if (!isVariableType(t.type())) {
    runtime_error("Expected object of type Variable but found type %s for argument #%d '%s'",
        t.type().toString(), pos, name);
  }
  return as_variable_ref(const_cast<Tensor&>(t));
}

Tensor & VariableType::unpack(const Tensor & t, const char * name, int pos) {
  return checked_cast_variable(t, name, pos).data();
}

SparseTensor VariableType::unpack(SparseTensor t, const char * name, int pos) {
  return SparseTensor(checked_cast_variable(t.tref, name, pos).data());
}

Tensor VariableType::unpack_opt(const Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    return Tensor();
  }
  return unpack(t, name, pos);
}

std::vector<at::Tensor> VariableType::unpack(at::TensorList tl, const char *name, int pos) {
  std::vector<at::Tensor> ret(tl.size());
  for (size_t i = 0; i < tl.size(); ++i) {
    const auto &t = tl[i];
    if (!t.defined()) {
      runtime_error("Expected a Tensor of type Variable but found an undefined Tensor at position #%d "
                    "for iterable argument #%d '%s'",
                    i, pos, name);
    }
    if (!isVariableType(t.type())) {
      runtime_error("Expected object of type Variable but found type %s at position #%d "
                    "for iterable argument #%d '%s'",
                    t.type().toString(), i, pos, name);
    }
    ret[i] = static_cast<const Variable&>(t).data();
  }
  return ret;
}

// Assumed that saved tensor lists are never inplace outputs
static std::vector<SavedVariable> make_saved_variable_list(TensorList tensors) {
  return fmap(tensors, [](const Tensor& tensor) -> SavedVariable {
      return SavedVariable{tensor, false /* is output */}; });
}

template <typename... Tensors, size_t... Is>
std::tuple<Tensors...> as_variable_impl(
    std::tuple<Tensors...> tensors,
    Indices<Is...>) {
  // Expand the integer parameter pack into a sequence of Variable
  // constructions. This turns into (boolean omitted):
  // Variable(std::get<0>(tensors)), Variable(std::get<1>(tensors)), ...
  return std::tuple<Tensors...>(
      make_variable(std::get<Is>(tensors), /*requires_grad=*/false)...);
}

template <typename... Tensors>
std::tuple<Tensors...> as_variable(std::tuple<Tensors...> tensors) {
  // `sizeof...(Tensors)` gets us the size of the `Tensors` parameter pack at
  // compile time. We use it to parameterize a `MakeIndices` class, which will
  // expand into an Indices object containing the numbers 0 to
  // sizeof...(Tensors) - 1.
  return as_variable_impl(
      tensors, typename MakeIndices<sizeof...(Tensors)>::indices());
}

static Tensor as_variable(Tensor tensor) {
  return make_variable(std::move(tensor), /*requires_grad=*/false);
}

static std::vector<Tensor> as_variable(TensorList tl) {
  std::vector<Tensor> variables;
  for (auto& t : tl) {
    variables.emplace_back(make_variable(std::move(t), /*requires_grad=*/false));
  }
  return variables;
}

static Tensor as_view(const Tensor & base, Tensor tensor) {
  auto base_var = Variable(base);
  if (base_var.is_view()) {
    base_var = base_var.base();
  }
  return make_variable_view(std::move(base_var), std::move(tensor));
}

#ifndef WITH_SCALARS
static void ensure_no_aten_scalars(Tensor & data) {
  if (data.defined() && data.dim() == 0) {
    data.as_strided_({1}, {1});
  }
}
#endif

struct ComputeRequiresGrad : IterArgs<ComputeRequiresGrad> {
  bool out = false;
  using IterArgs<ComputeRequiresGrad>::operator();
  void operator()(const at::Tensor& tensor) {
    const auto& var = static_cast<const Variable&>(tensor);
    if (var.defined() && var.requires_grad()) {
      out = true;
    }
  }
  bool short_circuit() { return out; }
};

template<typename... Args>
static bool compute_requires_grad(Args&&... args) {
  if (!GradMode::is_enabled()) {
    return false;
  }
  return ComputeRequiresGrad().apply(std::forward<Args>(args)...).out;
}

static void check_no_requires_grad(const Tensor& tensor, const char* name) {
  auto& var = static_cast<const Variable&>(tensor);
  if (var.defined() && var.requires_grad()) {
    std::string msg = "the derivative for '";
    msg += name;
    msg += "' is not implemented";
    throw std::runtime_error(msg);
  }
}

static void check_inplace(const Tensor& tensor) {
  auto& var = static_cast<const Variable&>(tensor);
  if (var.requires_grad() && var.is_leaf() && GradMode::is_enabled()) {
    at::runtime_error(
      "a leaf Variable that requires grad has been used in an in-place operation.");
  }
}

static void throw_error_out_requires_grad(const char* name) {
  at::runtime_error(
      "%s(): functions with out=... arguments don't support automatic differentiation, "
      "but one of the arguments requires grad.", name);
}

static void rebase_history(Tensor& tensor, std::shared_ptr<Function> grad_fn) {
  if (grad_fn && tensor.defined()) {
    auto& var = as_variable_ref(tensor);
    grad_fn->set_num_inputs(1);
    var.rebase_history({std::move(grad_fn), 0});
  }
}

static void rebase_history(TensorList tensors, std::shared_ptr<Function> grad_fn) {
  if (grad_fn) {
    grad_fn->set_num_inputs(tensors.size());
    uint32_t output_nr = 0;
    for (auto& tensor : tensors) {
      if (tensor.defined()) {
        auto& var = as_variable_ref(const_cast<Tensor&>(tensor));
        var.rebase_history({grad_fn, output_nr});
      }
      output_nr++;
    }
  }
}

// var must be the only differentiable output of the function. Use the ArrayRef
// overload for functions with multiple differentiable outputs.
static void set_history(Tensor& tensor, std::shared_ptr<Function> grad_fn) {
  if (grad_fn && tensor.defined()) {
    auto& var = as_variable_ref(tensor);
    autograd::create_gradient_edge(var, std::move(grad_fn));
  }
}

static void set_history(TensorList tensors, std::shared_ptr<Function> grad_fn) {
  if (grad_fn) {
    grad_fn->set_num_inputs(tensors.size());
    uint32_t output_nr = 0;
    for (auto& tensor : tensors) {
      if (tensor.defined()) {
        auto& var = as_variable_ref(const_cast<Tensor&>(tensor));
        var.set_gradient_edge({grad_fn, output_nr});
      }
      output_nr++;
    }
  }
}

struct Flatten : IterArgs<Flatten> {
  Flatten(variable_list& out) : out(out) {}
  variable_list& out;
  void operator()(const at::Tensor& x) { out.emplace_back(x); }
  void operator()(at::ArrayRef<at::Tensor> xs) {
    out.insert(out.end(), xs.begin(), xs.end());
  }
};

template<typename... Args> inline variable_list flatten(Args&&... args) {
  variable_list out;
  out.reserve(count_tensors(std::forward<Args>(args)...));
  Flatten(out).apply(std::forward<Args>(args)...);
  return out; // RVO
}

static void increment_version(Tensor & t) {
  as_variable_ref(t).bump_version();
}

static bool isFloatingPoint(ScalarType s) {
  return s == kFloat || s == kDouble || s == kHalf;
}

Tensor & VariableType::s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const {
  // TODO: once copy is exposed in Declarations.yaml we may be able to bind
  // it automatically
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  check_inplace(self);
  std::shared_ptr<CopyBackwards> grad_fn;
  auto requires_grad = compute_requires_grad(self, src);
  requires_grad &= isFloatingPoint(self.type().scalarType());
  if (requires_grad) {
    grad_fn = std::make_shared<CopyBackwards>();
    grad_fn->set_next_edges(collect_next_edges(self, src));
    grad_fn->set_num_inputs(1);
    grad_fn->src_type = &src.type();
    grad_fn->src_device = src.is_cuda() ? src.get_device() : -1;
  }
  baseType->s_copy_(self_, src_, non_blocking);
  increment_version(self);
  rebase_history(self, std::move(grad_fn));
  return self;
}

Tensor & VariableType::resize_(Tensor & self, IntList size) const {
  auto& self_ = unpack(self, "self", 0);
  if (as_variable_ref(self).requires_grad()) {
    at::runtime_error("cannot resize variables that require grad");
  }
  baseType->resize_(self_, size);
  return self;
}

Tensor & VariableType::resize_as_(Tensor & self, const Tensor & the_template) const {
  auto& self_ = unpack(self, "self", 0);
  auto& the_template_ = unpack(the_template, "the_template", 1);
  if (as_variable_ref(self).requires_grad()) {
    at::runtime_error("cannot resize variables that require grad");
  }
  baseType->resize_as_(self_, the_template_);
  return self;
}

Tensor VariableType::contiguous(const Tensor & self) const {
  unpack(self, "self", 0);
  if (self.is_contiguous()) {
    return self;
  }
  return self.clone();
}

static std::vector<int64_t> to_arg_sizes(TensorList tensors, int64_t dim) {
  std::vector<int64_t> arg_sizes(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    arg_sizes[i] = tensors[i].size(dim);
  }
  return arg_sizes;
}

int64_t VariableType::storage_offset(const Tensor & self) const {
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->storage_offset(self_);
  return result;
}
Tensor & VariableType::zeros_out(Tensor & result, IntList size) const {
  profiler::RecordFunction profiler("zeros_out");
  auto& result_ = unpack(result, "result", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result )) {
    trace_info = jit::tracer::preRecordTrace( "zeros_out", { result } );
    setattr(trace_info.n, jit::Symbol("size"), size);
  }
  baseType->zeros_out(result_, size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::zeros(IntList size) const {
  profiler::RecordFunction profiler("zeros");
  auto result = as_variable(baseType->zeros(size));
  return result;
}
Tensor & VariableType::ones_out(Tensor & result, IntList size) const {
  profiler::RecordFunction profiler("ones_out");
  auto& result_ = unpack(result, "result", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result )) {
    trace_info = jit::tracer::preRecordTrace( "ones_out", { result } );
    setattr(trace_info.n, jit::Symbol("size"), size);
  }
  baseType->ones_out(result_, size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::ones(IntList size) const {
  profiler::RecordFunction profiler("ones");
  auto result = as_variable(baseType->ones(size));
  return result;
}
int64_t VariableType::numel(const Tensor & self) const {
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->numel(self_);
  return result;
}
Tensor & VariableType::set_(Tensor & self, Storage & storage) const {
  profiler::RecordFunction profiler("set_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for set_ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  baseType->set_(self_, storage);
  increment_version(self);
  rebase_history(self, grad_fn);
  return self;
}
Tensor & VariableType::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
  profiler::RecordFunction profiler("set_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for set_ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  baseType->set_(self_, sourceStorage, storage_offset, size, stride);
  increment_version(self);
  rebase_history(self, grad_fn);
  return self;
}
Tensor & VariableType::set_(Tensor & self, const Tensor & source) const {
  profiler::RecordFunction profiler("set_");
  auto& self_ = unpack(self, "self", 0);
  auto& source_ = unpack(source, "source", 1);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::make_shared<Error>("the derivative for set_ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, source ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, source )) {
    trace_info = jit::tracer::preRecordTrace( "set", { self, source } );
  
  }
  baseType->set_(self_, source_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::set_(Tensor & self) const {
  profiler::RecordFunction profiler("set_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for set_ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "set", { self } );
  
  }
  baseType->set_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::fill_(Tensor & self, Scalar value) const {
  profiler::RecordFunction profiler("fill_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<FillBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<FillBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "fill", { self } );
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  baseType->fill_(self_, value);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::fill_(Tensor & self, const Tensor & value) const {
  profiler::RecordFunction profiler("fill_");
  auto& self_ = unpack(self, "self", 0);
  auto& value_ = unpack(value, "value", 1);
  check_inplace(self);
  std::shared_ptr<FillBackward1> grad_fn;
  if (compute_requires_grad( self, value )) {
    grad_fn = std::make_shared<FillBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, value ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, value )) {
    trace_info = jit::tracer::preRecordTrace( "fill", { self, value } );
  
  }
  baseType->fill_(self_, value_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
bool VariableType::is_contiguous(const Tensor & self) const {
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->is_contiguous(self_);
  return result;
}
bool VariableType::is_set_to(const Tensor & self, const Tensor & tensor) const {
  auto& self_ = unpack(self, "self", 0);
  auto& tensor_ = unpack(tensor, "tensor", 1);
  auto result = baseType->is_set_to(self_, tensor_);
  return result;
}
Tensor & VariableType::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
  profiler::RecordFunction profiler("masked_fill_");
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  check_inplace(self);
  std::shared_ptr<MaskedFillBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MaskedFillBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mask )) {
    trace_info = jit::tracer::preRecordTrace( "masked_fill", { self, mask } );
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  baseType->s_masked_fill_(self_, mask_, value);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
  profiler::RecordFunction profiler("masked_fill_");
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto& value_ = unpack(value, "value", 2);
  check_inplace(self);
  std::shared_ptr<MaskedFillBackward1> grad_fn;
  if (compute_requires_grad( self, value )) {
    grad_fn = std::make_shared<MaskedFillBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, value ));
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mask, value )) {
    trace_info = jit::tracer::preRecordTrace( "masked_fill", { self, mask, value } );
  
  }
  baseType->s_masked_fill_(self_, mask_, value_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
  profiler::RecordFunction profiler("masked_scatter_");
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto& source_ = unpack(source, "source", 2);
  check_inplace(self);
  std::shared_ptr<MaskedScatterBackward> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::make_shared<MaskedScatterBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->mask_ = SavedVariable(mask, false);
    grad_fn->source_sizes = source.sizes();
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mask, source )) {
    trace_info = jit::tracer::preRecordTrace( "masked_scatter", { self, mask, source } );
  
  }
  baseType->s_masked_scatter_(self_, mask_, source_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
  profiler::RecordFunction profiler("masked_select_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mask_ = unpack(mask, "mask", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("masked_select");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("masked_select");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, mask )) {
    trace_info = jit::tracer::preRecordTrace( "masked_select_out", { result, self, mask } );
  
  }
  baseType->s_masked_select_out(result_, self_, mask_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_masked_select(const Tensor & self, const Tensor & mask) const {
  profiler::RecordFunction profiler("masked_select");
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  std::shared_ptr<MaskedSelectBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MaskedSelectBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mask )) {
    trace_info = jit::tracer::preRecordTrace( "masked_select", { self, mask } );
  
  }
  auto result = as_variable(baseType->s_masked_select(self_, mask_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
  profiler::RecordFunction profiler("transpose");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TransposeBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TransposeBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim0 = dim0;
    grad_fn->dim1 = dim1;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "transpose", { self } );
    setattr(trace_info.n, jit::Symbol("dim0"), dim0);
    setattr(trace_info.n, jit::Symbol("dim1"), dim1);
  }
  auto result = as_view(self, baseType->transpose(self_, dim0, dim1));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::t(const Tensor & self) const {
  profiler::RecordFunction profiler("t");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "t", { self } );
  
  }
  auto result = as_view(self, baseType->t(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::nonzero_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("nonzero_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "nonzero_out", { result, self } );
  
  }
  baseType->nonzero_out(result_, self_);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::nonzero(const Tensor & self) const {
  profiler::RecordFunction profiler("nonzero");
  auto& self_ = unpack(self, "self", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "nonzero", { self } );
  
  }
  auto result = as_variable(baseType->nonzero(self_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::clone(const Tensor & self) const {
  profiler::RecordFunction profiler("clone");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CloneBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<CloneBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "clone", { self } );
  
  }
  auto result = as_variable(baseType->clone(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::view(const Tensor & self, IntList size) const {
  profiler::RecordFunction profiler("view");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ViewBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ViewBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "view", { self } );
    setattr(trace_info.n, jit::Symbol("size"), size);
  }
  auto result = as_view(self, baseType->view(self_, size));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
  profiler::RecordFunction profiler("index_select_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("index_select");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("index_select");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, index )) {
    trace_info = jit::tracer::preRecordTrace( "index_select_out", { result, self, index } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->index_select_out(result_, self_, dim, index_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
  profiler::RecordFunction profiler("index_select");
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  std::shared_ptr<IndexSelectBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<IndexSelectBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, index )) {
    trace_info = jit::tracer::preRecordTrace( "index_select", { self, index } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto result = as_variable(baseType->index_select(self_, dim, index_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
  profiler::RecordFunction profiler("index_copy_");
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& source_ = unpack(source, "source", 3);
  check_inplace(self);
  std::shared_ptr<IndexCopyBackward> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::make_shared<IndexCopyBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, index, source )) {
    trace_info = jit::tracer::preRecordTrace( "index_copy", { self, index, source } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->index_copy_(self_, dim, index_, source_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
  profiler::RecordFunction profiler("take_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("take");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("take");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, index )) {
    trace_info = jit::tracer::preRecordTrace( "take_out", { result, self, index } );
  
  }
  baseType->take_out(result_, self_, index_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::take(const Tensor & self, const Tensor & index) const {
  profiler::RecordFunction profiler("take");
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 1);
  std::shared_ptr<TakeBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TakeBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
    grad_fn->index_ = SavedVariable(index, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, index )) {
    trace_info = jit::tracer::preRecordTrace( "take", { self, index } );
  
  }
  auto result = as_variable(baseType->take(self_, index_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
  profiler::RecordFunction profiler("put_");
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 1);
  auto& source_ = unpack(source, "source", 2);
  check_inplace(self);
  std::shared_ptr<PutBackward> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::make_shared<PutBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->index_ = SavedVariable(index, false);
    grad_fn->source_info = source;
    grad_fn->accumulate = accumulate;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, index, source )) {
    trace_info = jit::tracer::preRecordTrace( "put", { self, index, source } );
    setattr(trace_info.n, jit::Symbol("accumulate"), accumulate);
  }
  baseType->put_(self_, index_, source_, accumulate);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
  profiler::RecordFunction profiler("index_add_");
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& source_ = unpack(source, "source", 3);
  check_inplace(self);
  std::shared_ptr<IndexAddBackward> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::make_shared<IndexAddBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, index, source )) {
    trace_info = jit::tracer::preRecordTrace( "index_add", { self, index, source } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->index_add_(self_, dim, index_, source_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
  profiler::RecordFunction profiler("index_fill_");
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  check_inplace(self);
  std::shared_ptr<IndexFillBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<IndexFillBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, index )) {
    trace_info = jit::tracer::preRecordTrace( "index_fill", { self, index } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  baseType->index_fill_(self_, dim, index_, value);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
  profiler::RecordFunction profiler("index_fill_");
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& value_ = unpack(value, "value", 3);
  check_inplace(self);
  std::shared_ptr<IndexFillBackward1> grad_fn;
  if (compute_requires_grad( self, value )) {
    grad_fn = std::make_shared<IndexFillBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, value ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, index, value )) {
    trace_info = jit::tracer::preRecordTrace( "index_fill", { self, index, value } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->index_fill_(self_, dim, index_, value_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor VariableType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
  profiler::RecordFunction profiler("unfold");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UnfoldBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<UnfoldBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dimension = dimension;
    grad_fn->size = size;
    grad_fn->step = step;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "unfold", { self } );
    setattr(trace_info.n, jit::Symbol("dimension"), dimension);
    setattr(trace_info.n, jit::Symbol("size"), size);
    setattr(trace_info.n, jit::Symbol("step"), step);
  }
  auto result = as_view(self, baseType->unfold(self_, dimension, size, step));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
  profiler::RecordFunction profiler("range_out");
  auto& result_ = unpack(result, "result", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result )) {
    trace_info = jit::tracer::preRecordTrace( "range_out", { result } );
    setattr(trace_info.n, jit::Symbol("start"), start);
    setattr(trace_info.n, jit::Symbol("end"), end);
    setattr(trace_info.n, jit::Symbol("step"), step);
  }
  baseType->range_out(result_, start, end, step);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::range(Scalar start, Scalar end, Scalar step) const {
  profiler::RecordFunction profiler("range");
  auto result = as_variable(baseType->range(start, end, step));
  return result;
}
Tensor & VariableType::arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
  profiler::RecordFunction profiler("arange_out");
  auto& result_ = unpack(result, "result", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result )) {
    trace_info = jit::tracer::preRecordTrace( "arange_out", { result } );
    setattr(trace_info.n, jit::Symbol("start"), start);
    setattr(trace_info.n, jit::Symbol("end"), end);
    setattr(trace_info.n, jit::Symbol("step"), step);
  }
  baseType->arange_out(result_, start, end, step);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::arange(Scalar start, Scalar end, Scalar step) const {
  profiler::RecordFunction profiler("arange");
  auto result = as_variable(baseType->arange(start, end, step));
  return result;
}
Tensor & VariableType::arange_out(Tensor & result, Scalar end) const {
  profiler::RecordFunction profiler("arange_out");
  auto& result_ = unpack(result, "result", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result )) {
    trace_info = jit::tracer::preRecordTrace( "arange_out", { result } );
    setattr(trace_info.n, jit::Symbol("end"), end);
  }
  baseType->arange_out(result_, end);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::arange(Scalar end) const {
  profiler::RecordFunction profiler("arange");
  auto result = as_variable(baseType->arange(end));
  return result;
}
Tensor & VariableType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
  profiler::RecordFunction profiler("scatter_");
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  check_inplace(self);
  std::shared_ptr<ScatterBackward0> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::make_shared<ScatterBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self, src ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, index, src )) {
    trace_info = jit::tracer::preRecordTrace( "scatter", { self, index, src } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->scatter_(self_, dim, index_, src_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
  profiler::RecordFunction profiler("scatter_");
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  check_inplace(self);
  std::shared_ptr<ScatterBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ScatterBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, index )) {
    trace_info = jit::tracer::preRecordTrace( "scatter", { self, index } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  baseType->scatter_(self_, dim, index_, value);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
  profiler::RecordFunction profiler("scatter_add_");
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  check_inplace(self);
  std::shared_ptr<ScatterAddBackward> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::make_shared<ScatterAddBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, src ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, index, src )) {
    trace_info = jit::tracer::preRecordTrace( "scatter_add", { self, index, src } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->scatter_add_(self_, dim, index_, src_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
  profiler::RecordFunction profiler("gather_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("gather");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("gather");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, index )) {
    trace_info = jit::tracer::preRecordTrace( "gather_out", { result, self, index } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->gather_out(result_, self_, dim, index_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
  profiler::RecordFunction profiler("gather");
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  std::shared_ptr<GatherBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<GatherBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, index )) {
    trace_info = jit::tracer::preRecordTrace( "gather", { self, index } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto result = as_variable(baseType->gather(self_, dim, index_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
void* VariableType::data_ptr(const Tensor & self) const {
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->data_ptr(self_);
  return result;
}
bool VariableType::equal(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("equal");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto result = baseType->equal(self_, other_);
  return result;
}
Tensor & VariableType::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__and___out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "__and___out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->__and___out(result_, self_, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::__and__(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__and__");
  auto& self_ = unpack(self, "self", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "__and_", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->__and__(self_, other));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__and___out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__and___out", { result, self, other } );
  
  }
  baseType->s___and___out(result_, self_, other_);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s___and__(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__and__");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__and_", { self, other } );
  
  }
  auto result = as_variable(baseType->s___and__(self_, other_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::__iand__(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__iand__");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for __iand__ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "__iand_", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->__iand__(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s___iand__(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__iand__");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<Error>("the derivative for __iand__ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__iand_", { self, other } );
  
  }
  baseType->s___iand__(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__or___out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "__or___out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->__or___out(result_, self_, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::__or__(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__or__");
  auto& self_ = unpack(self, "self", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "__or_", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->__or__(self_, other));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__or___out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__or___out", { result, self, other } );
  
  }
  baseType->s___or___out(result_, self_, other_);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s___or__(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__or__");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__or_", { self, other } );
  
  }
  auto result = as_variable(baseType->s___or__(self_, other_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::__ior__(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__ior__");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for __ior__ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "__ior_", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->__ior__(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s___ior__(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__ior__");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<Error>("the derivative for __ior__ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__ior_", { self, other } );
  
  }
  baseType->s___ior__(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__xor___out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "__xor___out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->__xor___out(result_, self_, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::__xor__(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__xor__");
  auto& self_ = unpack(self, "self", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "__xor_", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->__xor__(self_, other));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__xor___out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__xor___out", { result, self, other } );
  
  }
  baseType->s___xor___out(result_, self_, other_);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s___xor__(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__xor__");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__xor_", { self, other } );
  
  }
  auto result = as_variable(baseType->s___xor__(self_, other_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::__ixor__(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__ixor__");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for __ixor__ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "__ixor_", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->__ixor__(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s___ixor__(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__ixor__");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<Error>("the derivative for __ixor__ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__ixor_", { self, other } );
  
  }
  baseType->s___ixor__(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__lshift___out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "__lshift___out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->__lshift___out(result_, self_, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::__lshift__(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__lshift__");
  auto& self_ = unpack(self, "self", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "__lshift_", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->__lshift__(self_, other));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__lshift___out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__lshift___out", { result, self, other } );
  
  }
  baseType->s___lshift___out(result_, self_, other_);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s___lshift__(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__lshift__");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__lshift_", { self, other } );
  
  }
  auto result = as_variable(baseType->s___lshift__(self_, other_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::__ilshift__(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__ilshift__");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for __ilshift__ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "__ilshift_", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->__ilshift__(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s___ilshift__(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__ilshift__");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<Error>("the derivative for __ilshift__ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__ilshift_", { self, other } );
  
  }
  baseType->s___ilshift__(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__rshift___out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "__rshift___out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->__rshift___out(result_, self_, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::__rshift__(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__rshift__");
  auto& self_ = unpack(self, "self", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "__rshift_", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->__rshift__(self_, other));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__rshift___out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__rshift___out", { result, self, other } );
  
  }
  baseType->s___rshift___out(result_, self_, other_);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s___rshift__(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__rshift__");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__rshift_", { self, other } );
  
  }
  auto result = as_variable(baseType->s___rshift__(self_, other_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::__irshift__(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__irshift__");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for __irshift__ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "__irshift_", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->__irshift__(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s___irshift__(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__irshift__");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<Error>("the derivative for __irshift__ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "__irshift_", { self, other } );
  
  }
  baseType->s___irshift__(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("lt_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "lt_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->lt_out(result_, self_, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::lt(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("lt");
  auto& self_ = unpack(self, "self", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "lt", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->lt(self_, other));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("lt_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "lt_out", { result, self, other } );
  
  }
  baseType->s_lt_out(result_, self_, other_);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_lt(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("lt");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "lt", { self, other } );
  
  }
  auto result = as_variable(baseType->s_lt(self_, other_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::lt_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("lt_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LtBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<LtBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "lt", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->lt_(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_lt_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("lt_");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<LtBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<LtBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "lt", { self, other } );
  
  }
  baseType->s_lt_(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("gt_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "gt_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->gt_out(result_, self_, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::gt(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("gt");
  auto& self_ = unpack(self, "self", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "gt", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->gt(self_, other));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("gt_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "gt_out", { result, self, other } );
  
  }
  baseType->s_gt_out(result_, self_, other_);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_gt(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("gt");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "gt", { self, other } );
  
  }
  auto result = as_variable(baseType->s_gt(self_, other_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::gt_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("gt_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<GtBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<GtBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "gt", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->gt_(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_gt_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("gt_");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<GtBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<GtBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "gt", { self, other } );
  
  }
  baseType->s_gt_(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::le_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("le_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "le_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->le_out(result_, self_, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::le(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("le");
  auto& self_ = unpack(self, "self", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "le", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->le(self_, other));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("le_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "le_out", { result, self, other } );
  
  }
  baseType->s_le_out(result_, self_, other_);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_le(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("le");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "le", { self, other } );
  
  }
  auto result = as_variable(baseType->s_le(self_, other_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::le_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("le_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<LeBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "le", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->le_(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_le_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("le_");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<LeBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<LeBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "le", { self, other } );
  
  }
  baseType->s_le_(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("ge_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "ge_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->ge_out(result_, self_, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::ge(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("ge");
  auto& self_ = unpack(self, "self", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "ge", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->ge(self_, other));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("ge_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "ge_out", { result, self, other } );
  
  }
  baseType->s_ge_out(result_, self_, other_);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_ge(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("ge");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "ge", { self, other } );
  
  }
  auto result = as_variable(baseType->s_ge(self_, other_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::ge_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("ge_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<GeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<GeBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "ge", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->ge_(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_ge_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("ge_");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<GeBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<GeBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "ge", { self, other } );
  
  }
  baseType->s_ge_(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("eq_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "eq_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->eq_out(result_, self_, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::eq(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("eq");
  auto& self_ = unpack(self, "self", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "eq", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->eq(self_, other));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("eq_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "eq_out", { result, self, other } );
  
  }
  baseType->s_eq_out(result_, self_, other_);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_eq(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("eq");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "eq", { self, other } );
  
  }
  auto result = as_variable(baseType->s_eq(self_, other_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::eq_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("eq_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<EqBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<EqBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "eq", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->eq_(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_eq_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("eq_");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<EqBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<EqBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "eq", { self, other } );
  
  }
  baseType->s_eq_(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("ne_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "ne_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->ne_out(result_, self_, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::ne(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("ne");
  auto& self_ = unpack(self, "self", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "ne", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->ne(self_, other));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("ne_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "ne_out", { result, self, other } );
  
  }
  baseType->s_ne_out(result_, self_, other_);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_ne(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("ne");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "ne", { self, other } );
  
  }
  auto result = as_variable(baseType->s_ne(self_, other_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::ne_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("ne_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<NeBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "ne", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->ne_(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_ne_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("ne_");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<NeBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<NeBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "ne", { self, other } );
  
  }
  baseType->s_ne_(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
std::tuple<Tensor &,Tensor &> VariableType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("min_out");
  auto& min_ = unpack(min, "min", 0);
  auto& min_indices_ = unpack(min_indices, "min_indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("min");
  }
  if (compute_requires_grad( min )) {
    throw_error_out_requires_grad("min");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( min, min_indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "min_out", { min, min_indices, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  baseType->min_out(min_, min_indices_, self_, dim, keepdim);
  increment_version(min);
  rebase_history(min, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {min, min_indices} );
  }
  return std::forward_as_tuple(min, min_indices);
}
std::tuple<Tensor,Tensor> VariableType::min(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("min");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MinBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MinBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  Tensor min;
  Tensor min_indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "min", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  std::tie(min, min_indices) = as_variable(baseType->min(self_, dim, keepdim));
  set_history(min, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { min, min_indices } );
  }
  if (grad_fn) {
    grad_fn->min_indices_ = SavedVariable(min_indices, true);
  }
  return std::make_tuple(std::move(min), std::move(min_indices));
}
Tensor & VariableType::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("min_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("min");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("min");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "min_out", { result, self, other } );
  
  }
  baseType->s_min_out(result_, self_, other_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_min(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("min");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<MinBackward2> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<MinBackward2>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "min", { self, other } );
  
  }
  auto result = as_variable(baseType->s_min(self_, other_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::min(const Tensor & self) const {
  profiler::RecordFunction profiler("min");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MinBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MinBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "min", { self } );
  
  }
  auto result = as_variable(baseType->min(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("max_out");
  auto& max_ = unpack(max, "max", 0);
  auto& max_indices_ = unpack(max_indices, "max_indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max");
  }
  if (compute_requires_grad( max )) {
    throw_error_out_requires_grad("max");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( max, max_indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "max_out", { max, max_indices, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  baseType->max_out(max_, max_indices_, self_, dim, keepdim);
  increment_version(max);
  rebase_history(max, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {max, max_indices} );
  }
  return std::forward_as_tuple(max, max_indices);
}
std::tuple<Tensor,Tensor> VariableType::max(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("max");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MaxBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MaxBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  Tensor max;
  Tensor max_indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "max", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  std::tie(max, max_indices) = as_variable(baseType->max(self_, dim, keepdim));
  set_history(max, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { max, max_indices } );
  }
  if (grad_fn) {
    grad_fn->max_indices_ = SavedVariable(max_indices, true);
  }
  return std::make_tuple(std::move(max), std::move(max_indices));
}
Tensor & VariableType::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("max_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("max");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("max");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "max_out", { result, self, other } );
  
  }
  baseType->s_max_out(result_, self_, other_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_max(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("max");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<MaxBackward2> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<MaxBackward2>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "max", { self, other } );
  
  }
  auto result = as_variable(baseType->s_max(self_, other_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::max(const Tensor & self) const {
  profiler::RecordFunction profiler("max");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MaxBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MaxBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "max", { self } );
  
  }
  auto result = as_variable(baseType->max(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("kthvalue_out");
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("kthvalue");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("kthvalue");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( values, indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "kthvalue_out", { values, indices, self } );
    setattr(trace_info.n, jit::Symbol("k"), k);
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  baseType->kthvalue_out(values_, indices_, self_, k, dim, keepdim);
  increment_version(values);
  rebase_history(values, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {values, indices} );
  }
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor,Tensor> VariableType::kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("kthvalue");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<KthvalueBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<KthvalueBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  Tensor values;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "kthvalue", { self } );
    setattr(trace_info.n, jit::Symbol("k"), k);
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  std::tie(values, indices) = as_variable(baseType->kthvalue(self_, k, dim, keepdim));
  set_history(values, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { values, indices } );
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor &,Tensor &> VariableType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("mode_out");
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("mode");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("mode");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( values, indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "mode_out", { values, indices, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  baseType->mode_out(values_, indices_, self_, dim, keepdim);
  increment_version(values);
  rebase_history(values, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {values, indices} );
  }
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor,Tensor> VariableType::mode(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("mode");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ModeBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ModeBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  Tensor values;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "mode", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  std::tie(values, indices) = as_variable(baseType->mode(self_, dim, keepdim));
  set_history(values, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { values, indices } );
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor &,Tensor &> VariableType::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("median_out");
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("median");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("median");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( values, indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "median_out", { values, indices, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  baseType->median_out(values_, indices_, self_, dim, keepdim);
  increment_version(values);
  rebase_history(values, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {values, indices} );
  }
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor,Tensor> VariableType::median(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("median");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MedianBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MedianBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  Tensor values;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "median", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  std::tie(values, indices) = as_variable(baseType->median(self_, dim, keepdim));
  set_history(values, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { values, indices } );
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor VariableType::median(const Tensor & self) const {
  profiler::RecordFunction profiler("median");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MedianBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MedianBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "median", { self } );
  
  }
  auto result = as_variable(baseType->median(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
  profiler::RecordFunction profiler("sort_out");
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sort");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("sort");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( values, indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "sort_out", { values, indices, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("descending"), descending);
  }
  baseType->sort_out(values_, indices_, self_, dim, descending);
  increment_version(values);
  rebase_history(values, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {values, indices} );
  }
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor,Tensor> VariableType::sort(const Tensor & self, int64_t dim, bool descending) const {
  profiler::RecordFunction profiler("sort");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SortBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SortBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dim = dim;
  }
  Tensor values;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sort", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("descending"), descending);
  }
  std::tie(values, indices) = as_variable(baseType->sort(self_, dim, descending));
  set_history(values, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { values, indices } );
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor &,Tensor &> VariableType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
  profiler::RecordFunction profiler("topk_out");
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("topk");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("topk");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( values, indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "topk_out", { values, indices, self } );
    setattr(trace_info.n, jit::Symbol("k"), k);
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("largest"), largest);
    setattr(trace_info.n, jit::Symbol("sorted"), sorted);
  }
  baseType->topk_out(values_, indices_, self_, k, dim, largest, sorted);
  increment_version(values);
  rebase_history(values, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {values, indices} );
  }
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor,Tensor> VariableType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
  profiler::RecordFunction profiler("topk");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TopkBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TopkBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dim = dim;
  }
  Tensor values;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "topk", { self } );
    setattr(trace_info.n, jit::Symbol("k"), k);
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("largest"), largest);
    setattr(trace_info.n, jit::Symbol("sorted"), sorted);
  }
  std::tie(values, indices) = as_variable(baseType->topk(self_, k, dim, largest, sorted));
  set_history(values, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { values, indices } );
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor VariableType::all(const Tensor & self) const {
  profiler::RecordFunction profiler("all");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for all is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "all", { self } );
  
  }
  auto result = as_variable(baseType->all(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::any(const Tensor & self) const {
  profiler::RecordFunction profiler("any");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for any is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "any", { self } );
  
  }
  auto result = as_variable(baseType->any(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
int64_t VariableType::get_device(const Tensor & self) const {
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->get_device(self_);
  return result;
}
Tensor & VariableType::abs_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("abs_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("abs");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("abs");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "abs_out", { result, self } );
  
  }
  baseType->abs_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::abs(const Tensor & self) const {
  profiler::RecordFunction profiler("abs");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AbsBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AbsBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "abs", { self } );
  
  }
  auto result = as_variable(baseType->abs(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::abs_(Tensor & self) const {
  profiler::RecordFunction profiler("abs_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AbsBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AbsBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "abs", { self } );
  
  }
  baseType->abs_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::sigmoid_(Tensor & self) const {
  profiler::RecordFunction profiler("sigmoid_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SigmoidBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SigmoidBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sigmoid", { self } );
  
  }
  baseType->sigmoid_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::sigmoid_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("sigmoid_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sigmoid");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sigmoid");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "sigmoid_out", { result, self } );
  
  }
  baseType->sigmoid_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::sigmoid(const Tensor & self) const {
  profiler::RecordFunction profiler("sigmoid");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SigmoidBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SigmoidBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sigmoid", { self } );
  
  }
  auto result = as_variable(baseType->sigmoid(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::log_(Tensor & self) const {
  profiler::RecordFunction profiler("log_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LogBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<LogBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "log", { self } );
  
  }
  baseType->log_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::log_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("log_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("log");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("log");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "log_out", { result, self } );
  
  }
  baseType->log_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::log(const Tensor & self) const {
  profiler::RecordFunction profiler("log");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LogBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<LogBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "log", { self } );
  
  }
  auto result = as_variable(baseType->log(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::log1p_(Tensor & self) const {
  profiler::RecordFunction profiler("log1p_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Log1PBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Log1PBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "log1p", { self } );
  
  }
  baseType->log1p_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::log1p_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("log1p_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("log1p");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("log1p");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "log1p_out", { result, self } );
  
  }
  baseType->log1p_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::log1p(const Tensor & self) const {
  profiler::RecordFunction profiler("log1p");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Log1PBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Log1PBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "log1p", { self } );
  
  }
  auto result = as_variable(baseType->log1p(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::lgamma_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("lgamma_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("lgamma");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("lgamma");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "lgamma_out", { result, self } );
  
  }
  baseType->lgamma_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::lgamma(const Tensor & self) const {
  profiler::RecordFunction profiler("lgamma");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LgammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<LgammaBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "lgamma", { self } );
  
  }
  auto result = as_variable(baseType->lgamma(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::lgamma_(Tensor & self) const {
  profiler::RecordFunction profiler("lgamma_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LgammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<LgammaBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "lgamma", { self } );
  
  }
  baseType->lgamma_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::digamma_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("digamma_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("digamma");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("digamma");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "digamma_out", { result, self } );
  
  }
  baseType->digamma_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::digamma(const Tensor & self) const {
  profiler::RecordFunction profiler("digamma");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DigammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<DigammaBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "digamma", { self } );
  
  }
  auto result = as_variable(baseType->digamma(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::digamma_(Tensor & self) const {
  profiler::RecordFunction profiler("digamma_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<DigammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<DigammaBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "digamma", { self } );
  
  }
  baseType->digamma_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::polygamma_out(Tensor & result, int64_t n, const Tensor & self) const {
  profiler::RecordFunction profiler("polygamma_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("polygamma");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("polygamma");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "polygamma_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("n"), n);
  }
  baseType->polygamma_out(result_, n, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::polygamma(int64_t n, const Tensor & self) const {
  profiler::RecordFunction profiler("polygamma");
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<PolygammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<PolygammaBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->n = n;
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "polygamma", { self } );
    setattr(trace_info.n, jit::Symbol("n"), n);
  }
  auto result = as_variable(baseType->polygamma(n, self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::polygamma_(Tensor & self, int64_t n) const {
  profiler::RecordFunction profiler("polygamma_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for polygamma_ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "polygamma", { self } );
    setattr(trace_info.n, jit::Symbol("n"), n);
  }
  baseType->polygamma_(self_, n);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::exp_(Tensor & self) const {
  profiler::RecordFunction profiler("exp_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ExpBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ExpBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "exp", { self } );
  
  }
  baseType->exp_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::exp_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("exp_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("exp");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("exp");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "exp_out", { result, self } );
  
  }
  baseType->exp_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::exp(const Tensor & self) const {
  profiler::RecordFunction profiler("exp");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ExpBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ExpBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "exp", { self } );
  
  }
  auto result = as_variable(baseType->exp(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::expm1_(Tensor & self) const {
  profiler::RecordFunction profiler("expm1_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Expm1Backward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Expm1Backward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "expm1", { self } );
  
  }
  baseType->expm1_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::expm1_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("expm1_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("expm1");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("expm1");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "expm1_out", { result, self } );
  
  }
  baseType->expm1_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::expm1(const Tensor & self) const {
  profiler::RecordFunction profiler("expm1");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Expm1Backward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Expm1Backward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "expm1", { self } );
  
  }
  auto result = as_variable(baseType->expm1(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::cos_(Tensor & self) const {
  profiler::RecordFunction profiler("cos_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<CosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<CosBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "cos", { self } );
  
  }
  baseType->cos_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::cos_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("cos_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cos");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("cos");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "cos_out", { result, self } );
  
  }
  baseType->cos_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::cos(const Tensor & self) const {
  profiler::RecordFunction profiler("cos");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<CosBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "cos", { self } );
  
  }
  auto result = as_variable(baseType->cos(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::acos_(Tensor & self) const {
  profiler::RecordFunction profiler("acos_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AcosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AcosBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "acos", { self } );
  
  }
  baseType->acos_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::acos_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("acos_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("acos");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("acos");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "acos_out", { result, self } );
  
  }
  baseType->acos_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::acos(const Tensor & self) const {
  profiler::RecordFunction profiler("acos");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AcosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AcosBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "acos", { self } );
  
  }
  auto result = as_variable(baseType->acos(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::cosh_(Tensor & self) const {
  profiler::RecordFunction profiler("cosh_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<CoshBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<CoshBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "cosh", { self } );
  
  }
  baseType->cosh_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::cosh_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("cosh_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cosh");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("cosh");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "cosh_out", { result, self } );
  
  }
  baseType->cosh_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::cosh(const Tensor & self) const {
  profiler::RecordFunction profiler("cosh");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CoshBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<CoshBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "cosh", { self } );
  
  }
  auto result = as_variable(baseType->cosh(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::sin_(Tensor & self) const {
  profiler::RecordFunction profiler("sin_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SinBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sin", { self } );
  
  }
  baseType->sin_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::sin_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("sin_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sin");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sin");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "sin_out", { result, self } );
  
  }
  baseType->sin_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::sin(const Tensor & self) const {
  profiler::RecordFunction profiler("sin");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SinBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sin", { self } );
  
  }
  auto result = as_variable(baseType->sin(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::asin_(Tensor & self) const {
  profiler::RecordFunction profiler("asin_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AsinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AsinBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "asin", { self } );
  
  }
  baseType->asin_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::asin_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("asin_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("asin");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("asin");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "asin_out", { result, self } );
  
  }
  baseType->asin_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::asin(const Tensor & self) const {
  profiler::RecordFunction profiler("asin");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AsinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AsinBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "asin", { self } );
  
  }
  auto result = as_variable(baseType->asin(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::sinh_(Tensor & self) const {
  profiler::RecordFunction profiler("sinh_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SinhBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SinhBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sinh", { self } );
  
  }
  baseType->sinh_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::sinh_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("sinh_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sinh");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sinh");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "sinh_out", { result, self } );
  
  }
  baseType->sinh_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::sinh(const Tensor & self) const {
  profiler::RecordFunction profiler("sinh");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SinhBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SinhBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sinh", { self } );
  
  }
  auto result = as_variable(baseType->sinh(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::tan_(Tensor & self) const {
  profiler::RecordFunction profiler("tan_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TanBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "tan", { self } );
  
  }
  baseType->tan_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::tan_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("tan_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("tan");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("tan");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "tan_out", { result, self } );
  
  }
  baseType->tan_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::tan(const Tensor & self) const {
  profiler::RecordFunction profiler("tan");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TanBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "tan", { self } );
  
  }
  auto result = as_variable(baseType->tan(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::atan_(Tensor & self) const {
  profiler::RecordFunction profiler("atan_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AtanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AtanBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "atan", { self } );
  
  }
  baseType->atan_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::atan_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("atan_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("atan");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("atan");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "atan_out", { result, self } );
  
  }
  baseType->atan_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::atan(const Tensor & self) const {
  profiler::RecordFunction profiler("atan");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AtanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AtanBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "atan", { self } );
  
  }
  auto result = as_variable(baseType->atan(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::tanh_(Tensor & self) const {
  profiler::RecordFunction profiler("tanh_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TanhBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TanhBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "tanh", { self } );
  
  }
  baseType->tanh_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::tanh_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("tanh_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("tanh");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("tanh");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "tanh_out", { result, self } );
  
  }
  baseType->tanh_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::tanh(const Tensor & self) const {
  profiler::RecordFunction profiler("tanh");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TanhBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TanhBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "tanh", { self } );
  
  }
  auto result = as_variable(baseType->tanh(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::erf_(Tensor & self) const {
  profiler::RecordFunction profiler("erf_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ErfBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ErfBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "erf", { self } );
  
  }
  baseType->erf_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::erf_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("erf_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("erf");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("erf");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "erf_out", { result, self } );
  
  }
  baseType->erf_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::erf(const Tensor & self) const {
  profiler::RecordFunction profiler("erf");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ErfBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ErfBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "erf", { self } );
  
  }
  auto result = as_variable(baseType->erf(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::erfinv_(Tensor & self) const {
  profiler::RecordFunction profiler("erfinv_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ErfinvBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ErfinvBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "erfinv", { self } );
  
  }
  baseType->erfinv_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::erfinv_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("erfinv_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("erfinv");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("erfinv");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "erfinv_out", { result, self } );
  
  }
  baseType->erfinv_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::erfinv(const Tensor & self) const {
  profiler::RecordFunction profiler("erfinv");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ErfinvBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ErfinvBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "erfinv", { self } );
  
  }
  auto result = as_variable(baseType->erfinv(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::sqrt_(Tensor & self) const {
  profiler::RecordFunction profiler("sqrt_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SqrtBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SqrtBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sqrt", { self } );
  
  }
  baseType->sqrt_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::sqrt_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("sqrt_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sqrt");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sqrt");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "sqrt_out", { result, self } );
  
  }
  baseType->sqrt_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::sqrt(const Tensor & self) const {
  profiler::RecordFunction profiler("sqrt");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SqrtBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SqrtBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sqrt", { self } );
  
  }
  auto result = as_variable(baseType->sqrt(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::rsqrt_(Tensor & self) const {
  profiler::RecordFunction profiler("rsqrt_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<RsqrtBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RsqrtBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "rsqrt", { self } );
  
  }
  baseType->rsqrt_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::rsqrt_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("rsqrt_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("rsqrt");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("rsqrt");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "rsqrt_out", { result, self } );
  
  }
  baseType->rsqrt_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::rsqrt(const Tensor & self) const {
  profiler::RecordFunction profiler("rsqrt");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<RsqrtBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RsqrtBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "rsqrt", { self } );
  
  }
  auto result = as_variable(baseType->rsqrt(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::ceil_(Tensor & self) const {
  profiler::RecordFunction profiler("ceil_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<CeilBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<CeilBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "ceil", { self } );
  
  }
  baseType->ceil_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::ceil_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("ceil_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("ceil");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("ceil");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "ceil_out", { result, self } );
  
  }
  baseType->ceil_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::ceil(const Tensor & self) const {
  profiler::RecordFunction profiler("ceil");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CeilBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<CeilBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "ceil", { self } );
  
  }
  auto result = as_variable(baseType->ceil(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::floor_(Tensor & self) const {
  profiler::RecordFunction profiler("floor_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<FloorBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<FloorBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "floor", { self } );
  
  }
  baseType->floor_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::floor_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("floor_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("floor");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("floor");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "floor_out", { result, self } );
  
  }
  baseType->floor_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::floor(const Tensor & self) const {
  profiler::RecordFunction profiler("floor");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<FloorBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<FloorBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "floor", { self } );
  
  }
  auto result = as_variable(baseType->floor(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::round_(Tensor & self) const {
  profiler::RecordFunction profiler("round_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<RoundBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RoundBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "round", { self } );
  
  }
  baseType->round_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::round_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("round_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("round");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("round");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "round_out", { result, self } );
  
  }
  baseType->round_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::round(const Tensor & self) const {
  profiler::RecordFunction profiler("round");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<RoundBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RoundBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "round", { self } );
  
  }
  auto result = as_variable(baseType->round(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::trunc_(Tensor & self) const {
  profiler::RecordFunction profiler("trunc_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TruncBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TruncBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "trunc", { self } );
  
  }
  baseType->trunc_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::trunc_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("trunc_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("trunc");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("trunc");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "trunc_out", { result, self } );
  
  }
  baseType->trunc_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::trunc(const Tensor & self) const {
  profiler::RecordFunction profiler("trunc");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TruncBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TruncBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "trunc", { self } );
  
  }
  auto result = as_variable(baseType->trunc(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::frac_(Tensor & self) const {
  profiler::RecordFunction profiler("frac_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<FracBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<FracBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "frac", { self } );
  
  }
  baseType->frac_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::frac_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("frac_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("frac");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("frac");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "frac_out", { result, self } );
  
  }
  baseType->frac_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::frac(const Tensor & self) const {
  profiler::RecordFunction profiler("frac");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<FracBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<FracBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "frac", { self } );
  
  }
  auto result = as_variable(baseType->frac(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::mean_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("mean_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("mean");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("mean");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "mean_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  baseType->mean_out(result_, self_, dim, keepdim);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::mean(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("mean");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MeanBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MeanBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "mean", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  auto result = as_variable(baseType->mean(self_, dim, keepdim));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::mean(const Tensor & self) const {
  profiler::RecordFunction profiler("mean");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MeanBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MeanBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "mean", { self } );
  
  }
  auto result = as_variable(baseType->mean(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
  profiler::RecordFunction profiler("var_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("var");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("var");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "var_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("unbiased"), unbiased);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  baseType->var_out(result_, self_, dim, unbiased, keepdim);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
  profiler::RecordFunction profiler("var");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<VarBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<VarBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->unbiased = unbiased;
    grad_fn->keepdim = keepdim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "var", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("unbiased"), unbiased);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  auto result = as_variable(baseType->var(self_, dim, unbiased, keepdim));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::var(const Tensor & self, bool unbiased) const {
  profiler::RecordFunction profiler("var");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<VarBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<VarBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->unbiased = unbiased;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "var", { self } );
    setattr(trace_info.n, jit::Symbol("unbiased"), unbiased);
  }
  auto result = as_variable(baseType->var(self_, unbiased));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::std_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
  profiler::RecordFunction profiler("std_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("std");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("std");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "std_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("unbiased"), unbiased);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  baseType->std_out(result_, self_, dim, unbiased, keepdim);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
  profiler::RecordFunction profiler("std");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<StdBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<StdBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->unbiased = unbiased;
    grad_fn->keepdim = keepdim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "std", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("unbiased"), unbiased);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  auto result = as_variable(baseType->std(self_, dim, unbiased, keepdim));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor VariableType::std(const Tensor & self, bool unbiased) const {
  profiler::RecordFunction profiler("std");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<StdBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<StdBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->unbiased = unbiased;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "std", { self } );
    setattr(trace_info.n, jit::Symbol("unbiased"), unbiased);
  }
  auto result = as_variable(baseType->std(self_, unbiased));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::norm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("norm_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("norm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("norm");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "norm_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("p"), p);
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  baseType->norm_out(result_, self_, p, dim, keepdim);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("norm");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NormBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<NormBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "norm", { self } );
    setattr(trace_info.n, jit::Symbol("p"), p);
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  auto result = as_variable(baseType->norm(self_, p, dim, keepdim));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor VariableType::norm(const Tensor & self, Scalar p) const {
  profiler::RecordFunction profiler("norm");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NormBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<NormBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "norm", { self } );
    setattr(trace_info.n, jit::Symbol("p"), p);
  }
  auto result = as_variable(baseType->norm(self_, p));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
  profiler::RecordFunction profiler("renorm_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("renorm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("renorm");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "renorm_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("p"), p);
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("maxnorm"), maxnorm);
  }
  baseType->renorm_out(result_, self_, p, dim, maxnorm);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
  profiler::RecordFunction profiler("renorm");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<RenormBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RenormBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
    grad_fn->dim = dim;
    grad_fn->maxnorm = maxnorm;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "renorm", { self } );
    setattr(trace_info.n, jit::Symbol("p"), p);
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("maxnorm"), maxnorm);
  }
  auto result = as_variable(baseType->renorm(self_, p, dim, maxnorm));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
  profiler::RecordFunction profiler("renorm_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<RenormBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RenormBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->p = p;
    grad_fn->dim = dim;
    grad_fn->maxnorm = maxnorm;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "renorm", { self } );
    setattr(trace_info.n, jit::Symbol("p"), p);
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("maxnorm"), maxnorm);
  }
  baseType->renorm_(self_, p, dim, maxnorm);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor VariableType::s_dist(const Tensor & self, const Tensor & other, Scalar p) const {
  profiler::RecordFunction profiler("dist");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<DistBackward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<DistBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
    grad_fn->p = p;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "dist", { self, other } );
    setattr(trace_info.n, jit::Symbol("p"), p);
  }
  auto result = as_variable(baseType->s_dist(self_, other_, p));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::reciprocal_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("reciprocal_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("reciprocal");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("reciprocal");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "reciprocal_out", { result, self } );
  
  }
  baseType->reciprocal_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::reciprocal(const Tensor & self) const {
  profiler::RecordFunction profiler("reciprocal");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReciprocalBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ReciprocalBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "reciprocal", { self } );
  
  }
  auto result = as_variable(baseType->reciprocal(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::reciprocal_(Tensor & self) const {
  profiler::RecordFunction profiler("reciprocal_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ReciprocalBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ReciprocalBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "reciprocal", { self } );
  
  }
  baseType->reciprocal_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::neg_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("neg_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("neg");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("neg");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "neg_out", { result, self } );
  
  }
  baseType->neg_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::neg(const Tensor & self) const {
  profiler::RecordFunction profiler("neg");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NegBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<NegBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "neg", { self } );
  
  }
  auto result = as_variable(baseType->neg(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::neg_(Tensor & self) const {
  profiler::RecordFunction profiler("neg_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NegBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<NegBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "neg", { self } );
  
  }
  baseType->neg_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_atan2_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("atan2_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("atan2");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("atan2");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "atan2_out", { result, self, other } );
  
  }
  baseType->s_atan2_out(result_, self_, other_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_atan2(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("atan2");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<Atan2Backward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<Atan2Backward>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "atan2", { self, other } );
  
  }
  auto result = as_variable(baseType->s_atan2(self_, other_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_atan2_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("atan2_");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<Atan2Backward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<Atan2Backward>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "atan2", { self, other } );
  
  }
  baseType->s_atan2_(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
  profiler::RecordFunction profiler("pow_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("pow");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("pow");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "pow_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("exponent"), exponent);
  }
  baseType->pow_out(result_, self_, exponent);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::pow(const Tensor & self, Scalar exponent) const {
  profiler::RecordFunction profiler("pow");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PowBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<PowBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->exponent = exponent;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "pow", { self } );
    setattr(trace_info.n, jit::Symbol("exponent"), exponent);
  }
  auto result = as_variable(baseType->pow(self_, exponent));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
  profiler::RecordFunction profiler("pow_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& exponent_ = unpack(exponent, "exponent", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, exponent )) {
    throw_error_out_requires_grad("pow");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("pow");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, exponent )) {
    trace_info = jit::tracer::preRecordTrace( "pow_out", { result, self, exponent } );
  
  }
  baseType->s_pow_out(result_, self_, exponent_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_pow(const Tensor & self, const Tensor & exponent) const {
  profiler::RecordFunction profiler("pow");
  auto& self_ = unpack(self, "self", 0);
  auto& exponent_ = unpack(exponent, "exponent", 1);
  std::shared_ptr<PowBackward1> grad_fn;
  if (compute_requires_grad( self, exponent )) {
    grad_fn = std::make_shared<PowBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, exponent ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->exponent_ = SavedVariable(exponent, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, exponent )) {
    trace_info = jit::tracer::preRecordTrace( "pow", { self, exponent } );
  
  }
  auto result = as_variable(baseType->s_pow(self_, exponent_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::pow_out(Tensor & result, Scalar base, const Tensor & self) const {
  profiler::RecordFunction profiler("pow_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("pow");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("pow");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "pow_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("base"), base);
  }
  baseType->pow_out(result_, base, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::pow(Scalar base, const Tensor & self) const {
  profiler::RecordFunction profiler("pow");
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for pow is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "pow", { self } );
    setattr(trace_info.n, jit::Symbol("base"), base);
  }
  auto result = as_variable(baseType->pow(base, self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::pow_(Tensor & self, Scalar exponent) const {
  profiler::RecordFunction profiler("pow_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<PowBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<PowBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->exponent = exponent;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "pow", { self } );
    setattr(trace_info.n, jit::Symbol("exponent"), exponent);
  }
  baseType->pow_(self_, exponent);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_pow_(Tensor & self, const Tensor & exponent) const {
  profiler::RecordFunction profiler("pow_");
  auto& self_ = unpack(self, "self", 0);
  auto& exponent_ = unpack(exponent, "exponent", 1);
  check_inplace(self);
  std::shared_ptr<PowBackward1> grad_fn;
  if (compute_requires_grad( self, exponent )) {
    grad_fn = std::make_shared<PowBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, exponent ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->exponent_ = SavedVariable(exponent, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, exponent )) {
    trace_info = jit::tracer::preRecordTrace( "pow", { self, exponent } );
  
  }
  baseType->s_pow_(self_, exponent_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_lerp_out(Tensor & result, const Tensor & self, const Tensor & end, Scalar weight) const {
  profiler::RecordFunction profiler("lerp_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& end_ = unpack(end, "end", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, end )) {
    throw_error_out_requires_grad("lerp");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("lerp");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, end )) {
    trace_info = jit::tracer::preRecordTrace( "lerp_out", { result, self, end } );
    setattr(trace_info.n, jit::Symbol("weight"), weight);
  }
  baseType->s_lerp_out(result_, self_, end_, weight);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_lerp(const Tensor & self, const Tensor & end, Scalar weight) const {
  profiler::RecordFunction profiler("lerp");
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  std::shared_ptr<LerpBackward> grad_fn;
  if (compute_requires_grad( self, end )) {
    grad_fn = std::make_shared<LerpBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, end ));
    grad_fn->weight = weight;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, end )) {
    trace_info = jit::tracer::preRecordTrace( "lerp", { self, end } );
    setattr(trace_info.n, jit::Symbol("weight"), weight);
  }
  auto result = as_variable(baseType->s_lerp(self_, end_, weight));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_lerp_(Tensor & self, const Tensor & end, Scalar weight) const {
  profiler::RecordFunction profiler("lerp_");
  auto& self_ = unpack(self, "self", 0);
  auto& end_ = unpack(end, "end", 1);
  check_inplace(self);
  std::shared_ptr<LerpBackward> grad_fn;
  if (compute_requires_grad( self, end )) {
    grad_fn = std::make_shared<LerpBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, end ));
    grad_fn->weight = weight;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, end )) {
    trace_info = jit::tracer::preRecordTrace( "lerp", { self, end } );
    setattr(trace_info.n, jit::Symbol("weight"), weight);
  }
  baseType->s_lerp_(self_, end_, weight);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::linspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
  profiler::RecordFunction profiler("linspace_out");
  auto& result_ = unpack(result, "result", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result )) {
    trace_info = jit::tracer::preRecordTrace( "linspace_out", { result } );
    setattr(trace_info.n, jit::Symbol("start"), start);
    setattr(trace_info.n, jit::Symbol("end"), end);
    setattr(trace_info.n, jit::Symbol("steps"), steps);
  }
  baseType->linspace_out(result_, start, end, steps);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::linspace(Scalar start, Scalar end, int64_t steps) const {
  profiler::RecordFunction profiler("linspace");
  auto result = as_variable(baseType->linspace(start, end, steps));
  return result;
}
Tensor & VariableType::logspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
  profiler::RecordFunction profiler("logspace_out");
  auto& result_ = unpack(result, "result", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result )) {
    trace_info = jit::tracer::preRecordTrace( "logspace_out", { result } );
    setattr(trace_info.n, jit::Symbol("start"), start);
    setattr(trace_info.n, jit::Symbol("end"), end);
    setattr(trace_info.n, jit::Symbol("steps"), steps);
  }
  baseType->logspace_out(result_, start, end, steps);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::logspace(Scalar start, Scalar end, int64_t steps) const {
  profiler::RecordFunction profiler("logspace");
  auto result = as_variable(baseType->logspace(start, end, steps));
  return result;
}
Tensor & VariableType::histc_out(Tensor & result, const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
  profiler::RecordFunction profiler("histc_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("histc");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("histc");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "histc_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("bins"), bins);
    setattr(trace_info.n, jit::Symbol("min"), min);
    setattr(trace_info.n, jit::Symbol("max"), max);
  }
  baseType->histc_out(result_, self_, bins, min, max);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
  profiler::RecordFunction profiler("histc");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<HistcBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<HistcBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "histc", { self } );
    setattr(trace_info.n, jit::Symbol("bins"), bins);
    setattr(trace_info.n, jit::Symbol("min"), min);
    setattr(trace_info.n, jit::Symbol("max"), max);
  }
  auto result = as_variable(baseType->histc(self_, bins, min, max));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::zero_(Tensor & self) const {
  profiler::RecordFunction profiler("zero_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ZeroBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ZeroBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "zero", { self } );
  
  }
  baseType->zero_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("sum_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sum");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sum");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "sum_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  baseType->sum_out(result_, self_, dim, keepdim);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::sum(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("sum");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SumBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SumBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sum", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  auto result = as_variable(baseType->sum(self_, dim, keepdim));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::sum(const Tensor & self) const {
  profiler::RecordFunction profiler("sum");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SumBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SumBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sum", { self } );
  
  }
  auto result = as_variable(baseType->sum(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("prod_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("prod");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("prod");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "prod_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  baseType->prod_out(result_, self_, dim, keepdim);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::prod(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("prod");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ProdBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ProdBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "prod", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("keepdim"), keepdim);
  }
  auto result = as_variable(baseType->prod(self_, dim, keepdim));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor VariableType::prod(const Tensor & self) const {
  profiler::RecordFunction profiler("prod");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ProdBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ProdBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "prod", { self } );
  
  }
  auto result = as_variable(baseType->prod(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("cumsum_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cumsum");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("cumsum");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "cumsum_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->cumsum_out(result_, self_, dim);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::cumsum(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("cumsum");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CumsumBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<CumsumBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "cumsum", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto result = as_variable(baseType->cumsum(self_, dim));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("cumprod_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cumprod");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("cumprod");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "cumprod_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->cumprod_out(result_, self_, dim);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::cumprod(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("cumprod");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CumprodBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<CumprodBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "cumprod", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto result = as_variable(baseType->cumprod(self_, dim));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::sign_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("sign_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sign");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sign");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "sign_out", { result, self } );
  
  }
  baseType->sign_out(result_, self_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::sign(const Tensor & self) const {
  profiler::RecordFunction profiler("sign");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SignBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SignBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sign", { self } );
  
  }
  auto result = as_variable(baseType->sign(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::sign_(Tensor & self) const {
  profiler::RecordFunction profiler("sign_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SignBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SignBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sign", { self } );
  
  }
  baseType->sign_(self_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor VariableType::trace(const Tensor & self) const {
  profiler::RecordFunction profiler("trace");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TraceBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TraceBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "trace", { self } );
  
  }
  auto result = as_variable(baseType->trace(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
  profiler::RecordFunction profiler("add_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("add");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("add");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "add_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->add_out(result_, self_, other, alpha);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::add(const Tensor & self, Scalar other, Scalar alpha) const {
  profiler::RecordFunction profiler("add");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AddBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AddBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "add", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = as_variable(baseType->add(self_, other, alpha));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
  profiler::RecordFunction profiler("add_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("add");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("add");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "add_out", { result, self, other } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->s_add_out(result_, self_, other_, alpha);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
  profiler::RecordFunction profiler("add");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<AddBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<AddBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "add", { self, other } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = as_variable(baseType->s_add(self_, other_, alpha));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) const {
  profiler::RecordFunction profiler("add_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other.tref )) {
    throw_error_out_requires_grad("add");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("add");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "add_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->add_out(result_, self_, other_, alpha);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::add(const Tensor & self, SparseTensor other, Scalar alpha) const {
  profiler::RecordFunction profiler("add");
  auto& self_ = unpack(self, "self", 0);
  auto other_ = unpack(other, "other", 1);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, other.tref )) {
    grad_fn = std::make_shared<Error>("the derivative for add is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, other.tref ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "add", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = as_variable(baseType->add(self_, other_, alpha));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::add_(Tensor & self, Scalar other, Scalar alpha) const {
  profiler::RecordFunction profiler("add_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AddBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AddBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "add", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->add_(self_, other, alpha);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
  profiler::RecordFunction profiler("add_");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<AddBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<AddBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "add", { self, other } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->s_add_(self_, other_, alpha);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::add_(Tensor & self, SparseTensor other, Scalar alpha) const {
  profiler::RecordFunction profiler("add_");
  auto& self_ = unpack(self, "self", 0);
  auto other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, other.tref )) {
    grad_fn = std::make_shared<Error>("the derivative for add_ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, other.tref ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "add", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->add_(self_, other_, alpha);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
  profiler::RecordFunction profiler("sub_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sub");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sub");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "sub_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->sub_out(result_, self_, other, alpha);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
  profiler::RecordFunction profiler("sub");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SubBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SubBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sub", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = as_variable(baseType->sub(self_, other, alpha));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
  profiler::RecordFunction profiler("sub_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("sub");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sub");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "sub_out", { result, self, other } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->s_sub_out(result_, self_, other_, alpha);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
  profiler::RecordFunction profiler("sub");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<SubBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<SubBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "sub", { self, other } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = as_variable(baseType->s_sub(self_, other_, alpha));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
  profiler::RecordFunction profiler("sub_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SubBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SubBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sub", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->sub_(self_, other, alpha);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
  profiler::RecordFunction profiler("sub_");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<SubBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<SubBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "sub", { self, other } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->s_sub_(self_, other_, alpha);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("mul_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("mul");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("mul");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "mul_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->mul_out(result_, self_, other);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::mul(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("mul");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MulBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MulBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->other = other;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "mul", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->mul(self_, other));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("mul_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("mul");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("mul");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "mul_out", { result, self, other } );
  
  }
  baseType->s_mul_out(result_, self_, other_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_mul(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("mul");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<MulBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<MulBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "mul", { self, other } );
  
  }
  auto result = as_variable(baseType->s_mul(self_, other_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::mul_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("mul_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<MulBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MulBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->other = other;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "mul", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->mul_(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_mul_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("mul_");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<MulBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<MulBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "mul", { self, other } );
  
  }
  baseType->s_mul_(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("div_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("div");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("div");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "div_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->div_out(result_, self_, other);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::div(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("div");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DivBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<DivBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->other = other;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "div", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->div(self_, other));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("div_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("div");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("div");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "div_out", { result, self, other } );
  
  }
  baseType->s_div_out(result_, self_, other_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_div(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("div");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<DivBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<DivBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "div", { self, other } );
  
  }
  auto result = as_variable(baseType->s_div(self_, other_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::div_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("div_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<DivBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<DivBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->other = other;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "div", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->div_(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_div_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("div_");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<DivBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<DivBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "div", { self, other } );
  
  }
  baseType->s_div_(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("fmod_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("fmod");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("fmod");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "fmod_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->fmod_out(result_, self_, other);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::fmod(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("fmod");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<FmodBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<FmodBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "fmod", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->fmod(self_, other));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("fmod_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("fmod");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("fmod");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "fmod_out", { result, self, other } );
  
  }
  baseType->s_fmod_out(result_, self_, other_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_fmod(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("fmod");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<FmodBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<FmodBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_ = SavedVariable(other, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "fmod", { self, other } );
  
  }
  auto result = as_variable(baseType->s_fmod(self_, other_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::fmod_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("fmod_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<FmodBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<FmodBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "fmod", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->fmod_(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_fmod_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("fmod_");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<FmodBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<FmodBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_ = SavedVariable(other, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "fmod", { self, other } );
  
  }
  baseType->s_fmod_(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("remainder_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("remainder");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("remainder");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "remainder_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->remainder_out(result_, self_, other);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::remainder(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("remainder");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<RemainderBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RemainderBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "remainder", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  auto result = as_variable(baseType->remainder(self_, other));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("remainder_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("remainder");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("remainder");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "remainder_out", { result, self, other } );
  
  }
  baseType->s_remainder_out(result_, self_, other_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_remainder(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("remainder");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_no_requires_grad(other, "other");
  std::shared_ptr<RemainderBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RemainderBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "remainder", { self, other } );
  
  }
  auto result = as_variable(baseType->s_remainder(self_, other_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::remainder_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("remainder_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<RemainderBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RemainderBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "remainder", { self } );
    setattr(trace_info.n, jit::Symbol("other"), other);
  }
  baseType->remainder_(self_, other);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_remainder_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("remainder_");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  check_no_requires_grad(other, "other");
  std::shared_ptr<RemainderBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RemainderBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "remainder", { self, other } );
  
  }
  baseType->s_remainder_(self_, other_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
  profiler::RecordFunction profiler("clamp_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("clamp");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("clamp");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "clamp_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("min"), min);
    setattr(trace_info.n, jit::Symbol("max"), max);
  }
  baseType->clamp_out(result_, self_, min, max);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::clamp(const Tensor & self, Scalar min, Scalar max) const {
  profiler::RecordFunction profiler("clamp");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ClampBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ClampBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min = min;
    grad_fn->max = max;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "clamp", { self } );
    setattr(trace_info.n, jit::Symbol("min"), min);
    setattr(trace_info.n, jit::Symbol("max"), max);
  }
  auto result = as_variable(baseType->clamp(self_, min, max));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::clamp_(Tensor & self, Scalar min, Scalar max) const {
  profiler::RecordFunction profiler("clamp_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ClampBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ClampBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->min = min;
    grad_fn->max = max;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "clamp", { self } );
    setattr(trace_info.n, jit::Symbol("min"), min);
    setattr(trace_info.n, jit::Symbol("max"), max);
  }
  baseType->clamp_(self_, min, max);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
  profiler::RecordFunction profiler("clamp_min_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("clamp_min");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("clamp_min");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "clamp_min_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("min"), min);
  }
  baseType->clamp_min_out(result_, self_, min);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::clamp_min(const Tensor & self, Scalar min) const {
  profiler::RecordFunction profiler("clamp_min");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ClampMinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ClampMinBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min = min;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "clamp_min", { self } );
    setattr(trace_info.n, jit::Symbol("min"), min);
  }
  auto result = as_variable(baseType->clamp_min(self_, min));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::clamp_min_(Tensor & self, Scalar min) const {
  profiler::RecordFunction profiler("clamp_min_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ClampMinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ClampMinBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->min = min;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "clamp_min", { self } );
    setattr(trace_info.n, jit::Symbol("min"), min);
  }
  baseType->clamp_min_(self_, min);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
  profiler::RecordFunction profiler("clamp_max_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("clamp_max");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("clamp_max");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "clamp_max_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("max"), max);
  }
  baseType->clamp_max_out(result_, self_, max);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::clamp_max(const Tensor & self, Scalar max) const {
  profiler::RecordFunction profiler("clamp_max");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ClampMaxBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ClampMaxBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->max = max;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "clamp_max", { self } );
    setattr(trace_info.n, jit::Symbol("max"), max);
  }
  auto result = as_variable(baseType->clamp_max(self_, max));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::clamp_max_(Tensor & self, Scalar max) const {
  profiler::RecordFunction profiler("clamp_max_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ClampMaxBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ClampMaxBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->max = max;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "clamp_max", { self } );
    setattr(trace_info.n, jit::Symbol("max"), max);
  }
  baseType->clamp_max_(self_, max);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor VariableType::_dot(const Tensor & self, const Tensor & tensor) const {
  profiler::RecordFunction profiler("_dot");
  auto& self_ = unpack(self, "self", 0);
  auto& tensor_ = unpack(tensor, "tensor", 1);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, tensor )) {
    grad_fn = std::make_shared<Error>("the derivative for _dot is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, tensor ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, tensor )) {
    trace_info = jit::tracer::preRecordTrace( "_dot", { self, tensor } );
  
  }
  auto result = as_variable(baseType->_dot(self_, tensor_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("tril_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("tril");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("tril");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "tril_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("diagonal"), diagonal);
  }
  baseType->tril_out(result_, self_, diagonal);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::tril(const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("tril");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TrilBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TrilBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "tril", { self } );
    setattr(trace_info.n, jit::Symbol("diagonal"), diagonal);
  }
  auto result = as_variable(baseType->tril(self_, diagonal));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::tril_(Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("tril_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TrilBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TrilBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "tril", { self } );
    setattr(trace_info.n, jit::Symbol("diagonal"), diagonal);
  }
  baseType->tril_(self_, diagonal);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::triu_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("triu_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("triu");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("triu");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "triu_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("diagonal"), diagonal);
  }
  baseType->triu_out(result_, self_, diagonal);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::triu(const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("triu");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TriuBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TriuBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "triu", { self } );
    setattr(trace_info.n, jit::Symbol("diagonal"), diagonal);
  }
  auto result = as_variable(baseType->triu(self_, diagonal));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::triu_(Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("triu_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TriuBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TriuBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "triu", { self } );
    setattr(trace_info.n, jit::Symbol("diagonal"), diagonal);
  }
  baseType->triu_(self_, diagonal);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
  profiler::RecordFunction profiler("cross_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("cross");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("cross");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "cross_out", { result, self, other } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->cross_out(result_, self_, other_, dim);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
  profiler::RecordFunction profiler("cross");
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<CrossBackward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<CrossBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->other_ = SavedVariable(other, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "cross", { self, other } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto result = as_variable(baseType->cross(self_, other_, dim));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::eye_out(Tensor & result, int64_t n, int64_t m) const {
  profiler::RecordFunction profiler("eye_out");
  auto& result_ = unpack(result, "result", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result )) {
    trace_info = jit::tracer::preRecordTrace( "eye_out", { result } );
    setattr(trace_info.n, jit::Symbol("n"), n);
    setattr(trace_info.n, jit::Symbol("m"), m);
  }
  baseType->eye_out(result_, n, m);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::eye(int64_t n, int64_t m) const {
  profiler::RecordFunction profiler("eye");
  auto result = as_variable(baseType->eye(n, m));
  return result;
}
Tensor & VariableType::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("diag_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("diag");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("diag");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "diag_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("diagonal"), diagonal);
  }
  baseType->diag_out(result_, self_, diagonal);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::diag(const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("diag");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DiagBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<DiagBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->diagonal = diagonal;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "diag", { self } );
    setattr(trace_info.n, jit::Symbol("diagonal"), diagonal);
  }
  auto result = as_variable(baseType->diag(self_, diagonal));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmm_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat1_ = unpack(mat1, "mat1", 2);
  auto& mat2_ = unpack(mat2, "mat2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    throw_error_out_requires_grad("addmm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("addmm");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, mat1, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "addmm_out", { result, self, mat1, mat2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->s_addmm_out(result_, self_, mat1_, mat2_, beta, alpha);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmm");
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<AddmmBackward> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::make_shared<AddmmBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, mat1, mat2 ));
    grad_fn->mat1_sizes = mat1.sizes();
    grad_fn->mat1_ = SavedVariable(mat1, false);
    grad_fn->mat2_ = SavedVariable(mat2, false);
    grad_fn->alpha = alpha;
    grad_fn->mat2_sizes = mat2.sizes();
    grad_fn->beta = beta;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mat1, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "addmm", { self, mat1, mat2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = as_variable(baseType->s_addmm(self_, mat1_, mat2_, beta, alpha));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmm_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto mat1_ = unpack(mat1, "mat1", 2);
  auto& mat2_ = unpack(mat2, "mat2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat1.tref, mat2 )) {
    throw_error_out_requires_grad("addmm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("addmm");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "addmm_out", { result, self, mat2 } );
    setattr(trace_info.n, jit::Symbol("mat1"), mat1);
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->addmm_out(result_, self_, mat1_, mat2_, beta, alpha);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmm");
  auto& self_ = unpack(self, "self", 0);
  auto mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, mat1.tref, mat2 )) {
    grad_fn = std::make_shared<Error>("the derivative for addmm is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, mat1.tref, mat2 ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "addmm", { self, mat2 } );
    setattr(trace_info.n, jit::Symbol("mat1"), mat1);
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = as_variable(baseType->addmm(self_, mat1_, mat2_, beta, alpha));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmm_");
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  check_inplace(self);
  std::shared_ptr<AddmmBackward> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::make_shared<AddmmBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, mat1, mat2 ));
    grad_fn->mat1_sizes = mat1.sizes();
    grad_fn->mat1_ = SavedVariable(mat1, false);
    grad_fn->mat2_ = SavedVariable(mat2, false);
    grad_fn->alpha = alpha;
    grad_fn->mat2_sizes = mat2.sizes();
    grad_fn->beta = beta;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mat1, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "addmm", { self, mat1, mat2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->addmm_(self_, mat1_, mat2_, beta, alpha);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::addmm_(Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmm_");
  auto& self_ = unpack(self, "self", 0);
  auto mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, mat1.tref, mat2 )) {
    grad_fn = std::make_shared<Error>("the derivative for addmm_ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, mat1.tref, mat2 ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "addmm", { self, mat2 } );
    setattr(trace_info.n, jit::Symbol("mat1"), mat1);
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->addmm_(self_, mat1_, mat2_, beta, alpha);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s__addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_addmv_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat_ = unpack(mat, "mat", 2);
  auto& vec_ = unpack(vec, "vec", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat, vec )) {
    throw_error_out_requires_grad("_addmv");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_addmv");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, mat, vec )) {
    trace_info = jit::tracer::preRecordTrace( "_addmv_out", { result, self, mat, vec } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->s__addmv_out(result_, self_, mat_, vec_, beta, alpha);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_addmv");
  auto& self_ = unpack(self, "self", 0);
  auto& mat_ = unpack(mat, "mat", 1);
  auto& vec_ = unpack(vec, "vec", 2);
  std::shared_ptr<AddmvBackward> grad_fn;
  if (compute_requires_grad( self, mat, vec )) {
    grad_fn = std::make_shared<AddmvBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, mat, vec ));
    grad_fn->vec_ = SavedVariable(vec, false);
    grad_fn->alpha = alpha;
    grad_fn->beta = beta;
    grad_fn->mat_ = SavedVariable(mat, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mat, vec )) {
    trace_info = jit::tracer::preRecordTrace( "_addmv", { self, mat, vec } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = as_variable(baseType->s__addmv(self_, mat_, vec_, beta, alpha));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_addmv_");
  auto& self_ = unpack(self, "self", 0);
  auto& mat_ = unpack(mat, "mat", 1);
  auto& vec_ = unpack(vec, "vec", 2);
  check_inplace(self);
  std::shared_ptr<AddmvBackward> grad_fn;
  if (compute_requires_grad( self, mat, vec )) {
    grad_fn = std::make_shared<AddmvBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, mat, vec ));
    grad_fn->vec_ = SavedVariable(vec, false);
    grad_fn->alpha = alpha;
    grad_fn->beta = beta;
    grad_fn->mat_ = SavedVariable(mat, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mat, vec )) {
    trace_info = jit::tracer::preRecordTrace( "_addmv", { self, mat, vec } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->_addmv_(self_, mat_, vec_, beta, alpha);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s__addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_addr_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec1_ = unpack(vec1, "vec1", 2);
  auto& vec2_ = unpack(vec2, "vec2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, vec1, vec2 )) {
    throw_error_out_requires_grad("_addr");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_addr");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, vec1, vec2 )) {
    trace_info = jit::tracer::preRecordTrace( "_addr_out", { result, self, vec1, vec2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->s__addr_out(result_, self_, vec1_, vec2_, beta, alpha);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_addr");
  auto& self_ = unpack(self, "self", 0);
  auto& vec1_ = unpack(vec1, "vec1", 1);
  auto& vec2_ = unpack(vec2, "vec2", 2);
  std::shared_ptr<AddrBackward> grad_fn;
  if (compute_requires_grad( self, vec1, vec2 )) {
    grad_fn = std::make_shared<AddrBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, vec1, vec2 ));
    grad_fn->beta = beta;
    grad_fn->vec2_ = SavedVariable(vec2, false);
    grad_fn->alpha = alpha;
    grad_fn->vec1_ = SavedVariable(vec1, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, vec1, vec2 )) {
    trace_info = jit::tracer::preRecordTrace( "_addr", { self, vec1, vec2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = as_variable(baseType->s__addr(self_, vec1_, vec2_, beta, alpha));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_addr_");
  auto& self_ = unpack(self, "self", 0);
  auto& vec1_ = unpack(vec1, "vec1", 1);
  auto& vec2_ = unpack(vec2, "vec2", 2);
  check_inplace(self);
  std::shared_ptr<AddrBackward> grad_fn;
  if (compute_requires_grad( self, vec1, vec2 )) {
    grad_fn = std::make_shared<AddrBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, vec1, vec2 ));
    grad_fn->beta = beta;
    grad_fn->vec2_ = SavedVariable(vec2, false);
    grad_fn->alpha = alpha;
    grad_fn->vec1_ = SavedVariable(vec1, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, vec1, vec2 )) {
    trace_info = jit::tracer::preRecordTrace( "_addr", { self, vec1, vec2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->_addr_(self_, vec1_, vec2_, beta, alpha);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
  profiler::RecordFunction profiler("_ger_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec2_ = unpack(vec2, "vec2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, vec2 )) {
    throw_error_out_requires_grad("_ger");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_ger");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, vec2 )) {
    trace_info = jit::tracer::preRecordTrace( "_ger_out", { result, self, vec2 } );
  
  }
  baseType->_ger_out(result_, self_, vec2_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::_ger(const Tensor & self, const Tensor & vec2) const {
  profiler::RecordFunction profiler("_ger");
  auto& self_ = unpack(self, "self", 0);
  auto& vec2_ = unpack(vec2, "vec2", 1);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, vec2 )) {
    grad_fn = std::make_shared<Error>("the derivative for _ger is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, vec2 ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, vec2 )) {
    trace_info = jit::tracer::preRecordTrace( "_ger", { self, vec2 } );
  
  }
  auto result = as_variable(baseType->_ger(self_, vec2_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
  profiler::RecordFunction profiler("_mv_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec_ = unpack(vec, "vec", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, vec )) {
    throw_error_out_requires_grad("_mv");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_mv");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, vec )) {
    trace_info = jit::tracer::preRecordTrace( "_mv_out", { result, self, vec } );
  
  }
  baseType->_mv_out(result_, self_, vec_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::_mv(const Tensor & self, const Tensor & vec) const {
  profiler::RecordFunction profiler("_mv");
  auto& self_ = unpack(self, "self", 0);
  auto& vec_ = unpack(vec, "vec", 1);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, vec )) {
    grad_fn = std::make_shared<Error>("the derivative for _mv is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, vec ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, vec )) {
    trace_info = jit::tracer::preRecordTrace( "_mv", { self, vec } );
  
  }
  auto result = as_variable(baseType->_mv(self_, vec_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("_mm_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    throw_error_out_requires_grad("_mm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_mm");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "_mm_out", { result, self, mat2 } );
  
  }
  baseType->_mm_out(result_, self_, mat2_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::_mm(const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("_mm");
  auto& self_ = unpack(self, "self", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  std::shared_ptr<MmBackward> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    grad_fn = std::make_shared<MmBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, mat2 ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->mat2_sizes = mat2.sizes();
    grad_fn->mat2_ = SavedVariable(mat2, false);
    grad_fn->self_sizes = self.sizes();
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "_mm", { self, mat2 } );
  
  }
  auto result = as_variable(baseType->_mm(self_, mat2_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("bmm_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    throw_error_out_requires_grad("bmm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("bmm");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "bmm_out", { result, self, mat2 } );
  
  }
  baseType->bmm_out(result_, self_, mat2_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::bmm(const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("bmm");
  auto& self_ = unpack(self, "self", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  std::shared_ptr<BmmBackward> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    grad_fn = std::make_shared<BmmBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, mat2 ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->mat2_ = SavedVariable(mat2, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "bmm", { self, mat2 } );
  
  }
  auto result = as_variable(baseType->bmm(self_, mat2_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addbmm_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& batch1_ = unpack(batch1, "batch1", 2);
  auto& batch2_ = unpack(batch2, "batch2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    throw_error_out_requires_grad("addbmm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("addbmm");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, batch1, batch2 )) {
    trace_info = jit::tracer::preRecordTrace( "addbmm_out", { result, self, batch1, batch2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->s_addbmm_out(result_, self_, batch1_, batch2_, beta, alpha);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addbmm");
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  std::shared_ptr<AddbmmBackward> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    grad_fn = std::make_shared<AddbmmBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
    grad_fn->batch1_argsize_0 = batch1.size(0);
    grad_fn->batch1_argsize_1 = batch1.size(1);
    grad_fn->batch2_argsize_2 = batch2.size(2);
    grad_fn->batch2_ = SavedVariable(batch2, false);
    grad_fn->alpha = alpha;
    grad_fn->batch1_ = SavedVariable(batch1, false);
    grad_fn->beta = beta;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, batch1, batch2 )) {
    trace_info = jit::tracer::preRecordTrace( "addbmm", { self, batch1, batch2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = as_variable(baseType->s_addbmm(self_, batch1_, batch2_, beta, alpha));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addbmm_");
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  check_inplace(self);
  std::shared_ptr<AddbmmBackward> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    grad_fn = std::make_shared<AddbmmBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
    grad_fn->batch1_argsize_0 = batch1.size(0);
    grad_fn->batch1_argsize_1 = batch1.size(1);
    grad_fn->batch2_argsize_2 = batch2.size(2);
    grad_fn->batch2_ = SavedVariable(batch2, false);
    grad_fn->alpha = alpha;
    grad_fn->batch1_ = SavedVariable(batch1, false);
    grad_fn->beta = beta;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, batch1, batch2 )) {
    trace_info = jit::tracer::preRecordTrace( "addbmm", { self, batch1, batch2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->addbmm_(self_, batch1_, batch2_, beta, alpha);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("baddbmm_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& batch1_ = unpack(batch1, "batch1", 2);
  auto& batch2_ = unpack(batch2, "batch2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    throw_error_out_requires_grad("baddbmm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("baddbmm");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, batch1, batch2 )) {
    trace_info = jit::tracer::preRecordTrace( "baddbmm_out", { result, self, batch1, batch2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->s_baddbmm_out(result_, self_, batch1_, batch2_, beta, alpha);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("baddbmm");
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  std::shared_ptr<BaddbmmBackward> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    grad_fn = std::make_shared<BaddbmmBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
    grad_fn->batch2_ = SavedVariable(batch2, false);
    grad_fn->alpha = alpha;
    grad_fn->batch1_ = SavedVariable(batch1, false);
    grad_fn->beta = beta;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, batch1, batch2 )) {
    trace_info = jit::tracer::preRecordTrace( "baddbmm", { self, batch1, batch2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = as_variable(baseType->s_baddbmm(self_, batch1_, batch2_, beta, alpha));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("baddbmm_");
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  check_inplace(self);
  std::shared_ptr<BaddbmmBackward> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    grad_fn = std::make_shared<BaddbmmBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
    grad_fn->batch2_ = SavedVariable(batch2, false);
    grad_fn->alpha = alpha;
    grad_fn->batch1_ = SavedVariable(batch1, false);
    grad_fn->beta = beta;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, batch1, batch2 )) {
    trace_info = jit::tracer::preRecordTrace( "baddbmm", { self, batch1, batch2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->baddbmm_(self_, batch1_, batch2_, beta, alpha);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("addcmul_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& tensor1_ = unpack(tensor1, "tensor1", 2);
  auto& tensor2_ = unpack(tensor2, "tensor2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    throw_error_out_requires_grad("addcmul");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("addcmul");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, tensor1, tensor2 )) {
    trace_info = jit::tracer::preRecordTrace( "addcmul_out", { result, self, tensor1, tensor2 } );
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  baseType->s_addcmul_out(result_, self_, tensor1_, tensor2_, value);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("addcmul");
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  std::shared_ptr<AddcmulBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::make_shared<AddcmulBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    grad_fn->tensor2_ = SavedVariable(tensor2, false);
    grad_fn->value = value;
    grad_fn->tensor1_ = SavedVariable(tensor1, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, tensor1, tensor2 )) {
    trace_info = jit::tracer::preRecordTrace( "addcmul", { self, tensor1, tensor2 } );
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  auto result = as_variable(baseType->s_addcmul(self_, tensor1_, tensor2_, value));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("addcmul_");
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  check_inplace(self);
  std::shared_ptr<AddcmulBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::make_shared<AddcmulBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    grad_fn->tensor2_ = SavedVariable(tensor2, false);
    grad_fn->value = value;
    grad_fn->tensor1_ = SavedVariable(tensor1, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, tensor1, tensor2 )) {
    trace_info = jit::tracer::preRecordTrace( "addcmul", { self, tensor1, tensor2 } );
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  baseType->s_addcmul_(self_, tensor1_, tensor2_, value);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("addcdiv_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& tensor1_ = unpack(tensor1, "tensor1", 2);
  auto& tensor2_ = unpack(tensor2, "tensor2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    throw_error_out_requires_grad("addcdiv");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("addcdiv");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, tensor1, tensor2 )) {
    trace_info = jit::tracer::preRecordTrace( "addcdiv_out", { result, self, tensor1, tensor2 } );
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  baseType->s_addcdiv_out(result_, self_, tensor1_, tensor2_, value);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("addcdiv");
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  std::shared_ptr<AddcdivBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::make_shared<AddcdivBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    grad_fn->tensor2_ = SavedVariable(tensor2, false);
    grad_fn->value = value;
    grad_fn->tensor1_ = SavedVariable(tensor1, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, tensor1, tensor2 )) {
    trace_info = jit::tracer::preRecordTrace( "addcdiv", { self, tensor1, tensor2 } );
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  auto result = as_variable(baseType->s_addcdiv(self_, tensor1_, tensor2_, value));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("addcdiv_");
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  check_inplace(self);
  std::shared_ptr<AddcdivBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::make_shared<AddcdivBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    grad_fn->tensor2_ = SavedVariable(tensor2, false);
    grad_fn->value = value;
    grad_fn->tensor1_ = SavedVariable(tensor1, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, tensor1, tensor2 )) {
    trace_info = jit::tracer::preRecordTrace( "addcdiv", { self, tensor1, tensor2 } );
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  baseType->s_addcdiv_(self_, tensor1_, tensor2_, value);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
std::tuple<Tensor &,Tensor &> VariableType::gesv_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) const {
  profiler::RecordFunction profiler("gesv_out");
  auto& solution_ = unpack(solution, "solution", 0);
  auto& lu_ = unpack(lu, "lu", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& A_ = unpack(A, "A", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, A )) {
    throw_error_out_requires_grad("gesv");
  }
  if (compute_requires_grad( solution )) {
    throw_error_out_requires_grad("gesv");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( solution, lu, self, A )) {
    trace_info = jit::tracer::preRecordTrace( "gesv_out", { solution, lu, self, A } );
  
  }
  baseType->gesv_out(solution_, lu_, self_, A_);
  increment_version(solution);
  rebase_history(solution, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {solution, lu} );
  }
  return std::forward_as_tuple(solution, lu);
}
std::tuple<Tensor,Tensor> VariableType::gesv(const Tensor & self, const Tensor & A) const {
  profiler::RecordFunction profiler("gesv");
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<GesvBackward> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::make_shared<GesvBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, A ));
    grad_fn->A_ = SavedVariable(A, false);
  }
  Tensor solution;
  Tensor lu;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, A )) {
    trace_info = jit::tracer::preRecordTrace( "gesv", { self, A } );
  
  }
  std::tie(solution, lu) = as_variable(baseType->gesv(self_, A_));
  set_history(solution, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { solution, lu } );
  }
  if (grad_fn) {
    grad_fn->solution_ = SavedVariable(solution, true);
  }
  return std::make_tuple(std::move(solution), std::move(lu));
}
std::tuple<Tensor &,Tensor &> VariableType::gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A) const {
  profiler::RecordFunction profiler("gels_out");
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& A_ = unpack(A, "A", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, A )) {
    throw_error_out_requires_grad("gels");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("gels");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( res1, res2, self, A )) {
    trace_info = jit::tracer::preRecordTrace( "gels_out", { res1, res2, self, A } );
  
  }
  baseType->gels_out(res1_, res2_, self_, A_);
  increment_version(res1);
  increment_version(res2);
  rebase_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {res1, res2} );
  }
  return std::forward_as_tuple(res1, res2);
}
std::tuple<Tensor,Tensor> VariableType::gels(const Tensor & self, const Tensor & A) const {
  profiler::RecordFunction profiler("gels");
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<GelsBackward> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::make_shared<GelsBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, A ));
  }
  Tensor res1;
  Tensor res2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, A )) {
    trace_info = jit::tracer::preRecordTrace( "gels", { self, A } );
  
  }
  std::tie(res1, res2) = as_variable(baseType->gels(self_, A_));
  set_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { res1, res2 } );
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
std::tuple<Tensor &,Tensor &> VariableType::trtrs_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
  profiler::RecordFunction profiler("trtrs_out");
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& A_ = unpack(A, "A", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, A )) {
    throw_error_out_requires_grad("trtrs");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("trtrs");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( res1, res2, self, A )) {
    trace_info = jit::tracer::preRecordTrace( "trtrs_out", { res1, res2, self, A } );
    setattr(trace_info.n, jit::Symbol("upper"), upper);
    setattr(trace_info.n, jit::Symbol("transpose"), transpose);
    setattr(trace_info.n, jit::Symbol("unitriangular"), unitriangular);
  }
  baseType->trtrs_out(res1_, res2_, self_, A_, upper, transpose, unitriangular);
  increment_version(res1);
  increment_version(res2);
  rebase_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {res1, res2} );
  }
  return std::forward_as_tuple(res1, res2);
}
std::tuple<Tensor,Tensor> VariableType::trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
  profiler::RecordFunction profiler("trtrs");
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<TrtrsBackward> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::make_shared<TrtrsBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, A ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->A_ = SavedVariable(A, false);
    grad_fn->upper = upper;
    grad_fn->transpose = transpose;
    grad_fn->unitriangular = unitriangular;
  }
  Tensor res1;
  Tensor res2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, A )) {
    trace_info = jit::tracer::preRecordTrace( "trtrs", { self, A } );
    setattr(trace_info.n, jit::Symbol("upper"), upper);
    setattr(trace_info.n, jit::Symbol("transpose"), transpose);
    setattr(trace_info.n, jit::Symbol("unitriangular"), unitriangular);
  }
  std::tie(res1, res2) = as_variable(baseType->trtrs(self_, A_, upper, transpose, unitriangular));
  set_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { res1, res2 } );
  }
  if (grad_fn) {
    grad_fn->res1_ = SavedVariable(res1, true);
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
std::tuple<Tensor &,Tensor &> VariableType::symeig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors, bool upper) const {
  profiler::RecordFunction profiler("symeig_out");
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("symeig");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("symeig");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( res1, res2, self )) {
    trace_info = jit::tracer::preRecordTrace( "symeig_out", { res1, res2, self } );
    setattr(trace_info.n, jit::Symbol("eigenvectors"), eigenvectors);
    setattr(trace_info.n, jit::Symbol("upper"), upper);
  }
  baseType->symeig_out(res1_, res2_, self_, eigenvectors, upper);
  increment_version(res1);
  increment_version(res2);
  rebase_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {res1, res2} );
  }
  return std::forward_as_tuple(res1, res2);
}
std::tuple<Tensor,Tensor> VariableType::symeig(const Tensor & self, bool eigenvectors, bool upper) const {
  profiler::RecordFunction profiler("symeig");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SymeigBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SymeigBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor res1;
  Tensor res2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "symeig", { self } );
    setattr(trace_info.n, jit::Symbol("eigenvectors"), eigenvectors);
    setattr(trace_info.n, jit::Symbol("upper"), upper);
  }
  std::tie(res1, res2) = as_variable(baseType->symeig(self_, eigenvectors, upper));
  set_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { res1, res2 } );
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
std::tuple<Tensor &,Tensor &> VariableType::eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors) const {
  profiler::RecordFunction profiler("eig_out");
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("eig");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("eig");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( res1, res2, self )) {
    trace_info = jit::tracer::preRecordTrace( "eig_out", { res1, res2, self } );
    setattr(trace_info.n, jit::Symbol("eigenvectors"), eigenvectors);
  }
  baseType->eig_out(res1_, res2_, self_, eigenvectors);
  increment_version(res1);
  increment_version(res2);
  rebase_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {res1, res2} );
  }
  return std::forward_as_tuple(res1, res2);
}
std::tuple<Tensor,Tensor> VariableType::eig(const Tensor & self, bool eigenvectors) const {
  profiler::RecordFunction profiler("eig");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<EigBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<EigBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor res1;
  Tensor res2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "eig", { self } );
    setattr(trace_info.n, jit::Symbol("eigenvectors"), eigenvectors);
  }
  std::tie(res1, res2) = as_variable(baseType->eig(self_, eigenvectors));
  set_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { res1, res2 } );
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::svd_out(Tensor & res1, Tensor & res2, Tensor & res3, const Tensor & self, bool some) const {
  profiler::RecordFunction profiler("svd_out");
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& res3_ = unpack(res3, "res3", 2);
  auto& self_ = unpack(self, "self", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("svd");
  }
  if (compute_requires_grad( res1, res2, res3 )) {
    throw_error_out_requires_grad("svd");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( res1, res2, res3, self )) {
    trace_info = jit::tracer::preRecordTrace( "svd_out", { res1, res2, res3, self } );
    setattr(trace_info.n, jit::Symbol("some"), some);
  }
  baseType->svd_out(res1_, res2_, res3_, self_, some);
  increment_version(res1);
  increment_version(res2);
  increment_version(res3);
  rebase_history({ res1, res2, res3 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {res1, res2, res3} );
  }
  return std::forward_as_tuple(res1, res2, res3);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::svd(const Tensor & self, bool some) const {
  profiler::RecordFunction profiler("svd");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SvdBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SvdBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->some = some;
  }
  Tensor res1;
  Tensor res2;
  Tensor res3;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "svd", { self } );
    setattr(trace_info.n, jit::Symbol("some"), some);
  }
  std::tie(res1, res2, res3) = as_variable(baseType->svd(self_, some));
  set_history({ res1, res2, res3 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { res1, res2, res3 } );
  }
  if (grad_fn) {
    grad_fn->res1_ = SavedVariable(res1, true);
    grad_fn->res2_ = SavedVariable(res2, true);
    grad_fn->res3_ = SavedVariable(res3, true);
  }
  return std::make_tuple(std::move(res1), std::move(res2), std::move(res3));
}
Tensor & VariableType::inverse_out(Tensor & output, const Tensor & self) const {
  profiler::RecordFunction profiler("inverse_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("inverse");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("inverse");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "inverse_out", { output, self } );
  
  }
  baseType->inverse_out(output_, self_);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::inverse(const Tensor & self) const {
  profiler::RecordFunction profiler("inverse");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<InverseBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<InverseBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "inverse", { self } );
  
  }
  auto output = as_variable(baseType->inverse(self_));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  if (grad_fn) {
    grad_fn->output_ = SavedVariable(output, true);
  }
  return output;
}
Tensor & VariableType::potrf_out(Tensor & output, const Tensor & self, bool upper) const {
  profiler::RecordFunction profiler("potrf_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("potrf");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("potrf");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "potrf_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("upper"), upper);
  }
  baseType->potrf_out(output_, self_, upper);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::potrf(const Tensor & self, bool upper) const {
  profiler::RecordFunction profiler("potrf");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PotrfBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<PotrfBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->upper = upper;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "potrf", { self } );
    setattr(trace_info.n, jit::Symbol("upper"), upper);
  }
  auto output = as_variable(baseType->potrf(self_, upper));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  if (grad_fn) {
    grad_fn->output_ = SavedVariable(output, true);
  }
  return output;
}
Tensor & VariableType::potrs_out(Tensor & result, const Tensor & self, const Tensor & input2, bool upper) const {
  profiler::RecordFunction profiler("potrs_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& input2_ = unpack(input2, "input2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    throw_error_out_requires_grad("potrs");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("potrs");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, input2 )) {
    trace_info = jit::tracer::preRecordTrace( "potrs_out", { result, self, input2 } );
    setattr(trace_info.n, jit::Symbol("upper"), upper);
  }
  baseType->potrs_out(result_, self_, input2_, upper);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::potrs(const Tensor & self, const Tensor & input2, bool upper) const {
  profiler::RecordFunction profiler("potrs");
  auto& self_ = unpack(self, "self", 0);
  auto& input2_ = unpack(input2, "input2", 1);
  std::shared_ptr<PotrsBackward> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    grad_fn = std::make_shared<PotrsBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, input2 ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, input2 )) {
    trace_info = jit::tracer::preRecordTrace( "potrs", { self, input2 } );
    setattr(trace_info.n, jit::Symbol("upper"), upper);
  }
  auto result = as_variable(baseType->potrs(self_, input2_, upper));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::potri_out(Tensor & output, const Tensor & self, bool upper) const {
  profiler::RecordFunction profiler("potri_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("potri");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("potri");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "potri_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("upper"), upper);
  }
  baseType->potri_out(output_, self_, upper);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::potri(const Tensor & self, bool upper) const {
  profiler::RecordFunction profiler("potri");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PotriBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<PotriBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "potri", { self } );
    setattr(trace_info.n, jit::Symbol("upper"), upper);
  }
  auto output = as_variable(baseType->potri(self_, upper));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &> VariableType::pstrf_out(Tensor & res1, Tensor & res2, const Tensor & self, bool upper, Scalar tol) const {
  profiler::RecordFunction profiler("pstrf_out");
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("pstrf");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("pstrf");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( res1, res2, self )) {
    trace_info = jit::tracer::preRecordTrace( "pstrf_out", { res1, res2, self } );
    setattr(trace_info.n, jit::Symbol("upper"), upper);
    setattr(trace_info.n, jit::Symbol("tol"), tol);
  }
  baseType->pstrf_out(res1_, res2_, self_, upper, tol);
  increment_version(res1);
  increment_version(res2);
  rebase_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {res1, res2} );
  }
  return std::forward_as_tuple(res1, res2);
}
std::tuple<Tensor,Tensor> VariableType::pstrf(const Tensor & self, bool upper, Scalar tol) const {
  profiler::RecordFunction profiler("pstrf");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PstrfBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<PstrfBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor res1;
  Tensor res2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "pstrf", { self } );
    setattr(trace_info.n, jit::Symbol("upper"), upper);
    setattr(trace_info.n, jit::Symbol("tol"), tol);
  }
  std::tie(res1, res2) = as_variable(baseType->pstrf(self_, upper, tol));
  set_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { res1, res2 } );
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
std::tuple<Tensor &,Tensor &> VariableType::qr_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
  profiler::RecordFunction profiler("qr_out");
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("qr");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("qr");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( res1, res2, self )) {
    trace_info = jit::tracer::preRecordTrace( "qr_out", { res1, res2, self } );
  
  }
  baseType->qr_out(res1_, res2_, self_);
  increment_version(res1);
  increment_version(res2);
  rebase_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {res1, res2} );
  }
  return std::forward_as_tuple(res1, res2);
}
std::tuple<Tensor,Tensor> VariableType::qr(const Tensor & self) const {
  profiler::RecordFunction profiler("qr");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<QrBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<QrBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor res1;
  Tensor res2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "qr", { self } );
  
  }
  std::tie(res1, res2) = as_variable(baseType->qr(self_));
  set_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { res1, res2 } );
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
std::tuple<Tensor &,Tensor &> VariableType::geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
  profiler::RecordFunction profiler("geqrf_out");
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("geqrf");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("geqrf");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( res1, res2, self )) {
    trace_info = jit::tracer::preRecordTrace( "geqrf_out", { res1, res2, self } );
  
  }
  baseType->geqrf_out(res1_, res2_, self_);
  increment_version(res1);
  increment_version(res2);
  rebase_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {res1, res2} );
  }
  return std::forward_as_tuple(res1, res2);
}
std::tuple<Tensor,Tensor> VariableType::geqrf(const Tensor & self) const {
  profiler::RecordFunction profiler("geqrf");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<GeqrfBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<GeqrfBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor res1;
  Tensor res2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "geqrf", { self } );
  
  }
  std::tie(res1, res2) = as_variable(baseType->geqrf(self_));
  set_history({ res1, res2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { res1, res2 } );
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
Tensor & VariableType::orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2) const {
  profiler::RecordFunction profiler("orgqr_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& input2_ = unpack(input2, "input2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    throw_error_out_requires_grad("orgqr");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("orgqr");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, input2 )) {
    trace_info = jit::tracer::preRecordTrace( "orgqr_out", { result, self, input2 } );
  
  }
  baseType->orgqr_out(result_, self_, input2_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::orgqr(const Tensor & self, const Tensor & input2) const {
  profiler::RecordFunction profiler("orgqr");
  auto& self_ = unpack(self, "self", 0);
  auto& input2_ = unpack(input2, "input2", 1);
  std::shared_ptr<OrgqrBackward> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    grad_fn = std::make_shared<OrgqrBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, input2 ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, input2 )) {
    trace_info = jit::tracer::preRecordTrace( "orgqr", { self, input2 } );
  
  }
  auto result = as_variable(baseType->orgqr(self_, input2_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
  profiler::RecordFunction profiler("ormqr_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& input2_ = unpack(input2, "input2", 2);
  auto& input3_ = unpack(input3, "input3", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, input2, input3 )) {
    throw_error_out_requires_grad("ormqr");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("ormqr");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, input2, input3 )) {
    trace_info = jit::tracer::preRecordTrace( "ormqr_out", { result, self, input2, input3 } );
    setattr(trace_info.n, jit::Symbol("left"), left);
    setattr(trace_info.n, jit::Symbol("transpose"), transpose);
  }
  baseType->ormqr_out(result_, self_, input2_, input3_, left, transpose);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
  profiler::RecordFunction profiler("ormqr");
  auto& self_ = unpack(self, "self", 0);
  auto& input2_ = unpack(input2, "input2", 1);
  auto& input3_ = unpack(input3, "input3", 2);
  std::shared_ptr<OrmqrBackward> grad_fn;
  if (compute_requires_grad( self, input2, input3 )) {
    grad_fn = std::make_shared<OrmqrBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, input2, input3 ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, input2, input3 )) {
    trace_info = jit::tracer::preRecordTrace( "ormqr", { self, input2, input3 } );
    setattr(trace_info.n, jit::Symbol("left"), left);
    setattr(trace_info.n, jit::Symbol("transpose"), transpose);
  }
  auto result = as_variable(baseType->ormqr(self_, input2_, input3_, left, transpose));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::btrifact_out(Tensor & result, Tensor & pivots, const Tensor & self, bool pivot) const {
  profiler::RecordFunction profiler("btrifact_out");
  auto& result_ = unpack(result, "result", 0);
  auto& pivots_ = unpack(pivots, "pivots", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("btrifact");
  }
  if (compute_requires_grad( result, pivots )) {
    throw_error_out_requires_grad("btrifact");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, pivots, self )) {
    trace_info = jit::tracer::preRecordTrace( "btrifact_out", { result, pivots, self } );
    setattr(trace_info.n, jit::Symbol("pivot"), pivot);
  }
  baseType->btrifact_out(result_, pivots_, self_, pivot);
  increment_version(result);
  increment_version(pivots);
  rebase_history({ result, pivots }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result, pivots} );
  }
  return std::forward_as_tuple(result, pivots);
}
std::tuple<Tensor,Tensor> VariableType::btrifact(const Tensor & self, bool pivot) const {
  profiler::RecordFunction profiler("btrifact");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<BtrifactBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<BtrifactBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result;
  Tensor pivots;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "btrifact", { self } );
    setattr(trace_info.n, jit::Symbol("pivot"), pivot);
  }
  std::tie(result, pivots) = as_variable(baseType->btrifact(self_, pivot));
  set_history({ result, pivots }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result, pivots } );
  }
  return std::make_tuple(std::move(result), std::move(pivots));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::btrifact_with_info_out(Tensor & result, Tensor & pivots, Tensor & info, const Tensor & self, bool pivot) const {
  profiler::RecordFunction profiler("btrifact_with_info_out");
  auto& result_ = unpack(result, "result", 0);
  auto& pivots_ = unpack(pivots, "pivots", 1);
  auto& info_ = unpack(info, "info", 2);
  auto& self_ = unpack(self, "self", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("btrifact_with_info");
  }
  if (compute_requires_grad( result, pivots, info )) {
    throw_error_out_requires_grad("btrifact_with_info");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, pivots, info, self )) {
    trace_info = jit::tracer::preRecordTrace( "btrifact_with_info_out", { result, pivots, info, self } );
    setattr(trace_info.n, jit::Symbol("pivot"), pivot);
  }
  baseType->btrifact_with_info_out(result_, pivots_, info_, self_, pivot);
  increment_version(result);
  increment_version(pivots);
  increment_version(info);
  rebase_history({ result, pivots, info }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result, pivots, info} );
  }
  return std::forward_as_tuple(result, pivots, info);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::btrifact_with_info(const Tensor & self, bool pivot) const {
  profiler::RecordFunction profiler("btrifact_with_info");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<BtrifactWithInfoBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<BtrifactWithInfoBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result;
  Tensor pivots;
  Tensor info;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "btrifact_with_info", { self } );
    setattr(trace_info.n, jit::Symbol("pivot"), pivot);
  }
  std::tie(result, pivots, info) = as_variable(baseType->btrifact_with_info(self_, pivot));
  set_history({ result, pivots, info }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result, pivots, info } );
  }
  return std::make_tuple(std::move(result), std::move(pivots), std::move(info));
}
Tensor & VariableType::btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
  profiler::RecordFunction profiler("btrisolve_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& LU_data_ = unpack(LU_data, "LU_data", 2);
  auto& LU_pivots_ = unpack(LU_pivots, "LU_pivots", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, LU_data, LU_pivots )) {
    throw_error_out_requires_grad("btrisolve");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("btrisolve");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, LU_data, LU_pivots )) {
    trace_info = jit::tracer::preRecordTrace( "btrisolve_out", { result, self, LU_data, LU_pivots } );
  
  }
  baseType->btrisolve_out(result_, self_, LU_data_, LU_pivots_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
  profiler::RecordFunction profiler("btrisolve");
  auto& self_ = unpack(self, "self", 0);
  auto& LU_data_ = unpack(LU_data, "LU_data", 1);
  auto& LU_pivots_ = unpack(LU_pivots, "LU_pivots", 2);
  check_no_requires_grad(LU_data, "LU_data");
  check_no_requires_grad(LU_pivots, "LU_pivots");
  std::shared_ptr<BtrisolveBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<BtrisolveBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, LU_data, LU_pivots )) {
    trace_info = jit::tracer::preRecordTrace( "btrisolve", { self, LU_data, LU_pivots } );
  
  }
  auto result = as_variable(baseType->btrisolve(self_, LU_data_, LU_pivots_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::randperm_out(Tensor & result, int64_t n, Generator * generator) const {
  profiler::RecordFunction profiler("randperm_out");
  auto& result_ = unpack(result, "result", 0);
  baseType->randperm_out(result_, n, generator);
  return result;
}
Tensor VariableType::randperm(int64_t n, Generator * generator) const {
  profiler::RecordFunction profiler("randperm");
  auto result = as_variable(baseType->randperm(n, generator));
  return result;
}
Tensor & VariableType::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
  profiler::RecordFunction profiler("random_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<RandomBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RandomBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  baseType->random_(self_, from, to, generator);
  increment_version(self);
  rebase_history(self, grad_fn);
  return self;
}
Tensor & VariableType::random_(Tensor & self, int64_t to, Generator * generator) const {
  profiler::RecordFunction profiler("random_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<RandomBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RandomBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  baseType->random_(self_, to, generator);
  increment_version(self);
  rebase_history(self, grad_fn);
  return self;
}
Tensor & VariableType::random_(Tensor & self, Generator * generator) const {
  profiler::RecordFunction profiler("random_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<RandomBackward2> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RandomBackward2>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  baseType->random_(self_, generator);
  increment_version(self);
  rebase_history(self, grad_fn);
  return self;
}
Tensor & VariableType::multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
  profiler::RecordFunction profiler("multinomial_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  baseType->multinomial_out(result_, self_, num_samples, replacement, generator);
  return result;
}
Tensor VariableType::multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
  profiler::RecordFunction profiler("multinomial");
  auto& self_ = unpack(self, "self", 0);
  auto result = as_variable(baseType->multinomial(self_, num_samples, replacement, generator));
  return result;
}
Tensor & VariableType::uniform_(Tensor & self, double from, double to, Generator * generator) const {
  profiler::RecordFunction profiler("uniform_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<UniformBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<UniformBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  baseType->uniform_(self_, from, to, generator);
  increment_version(self);
  rebase_history(self, grad_fn);
  return self;
}
Tensor & VariableType::normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator) const {
  profiler::RecordFunction profiler("normal_out");
  auto& output_ = unpack(output, "output", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( mean )) {
    throw_error_out_requires_grad("normal");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("normal");
  }
  baseType->normal_out(output_, mean_, std, generator);
  increment_version(output);
  rebase_history(output, grad_fn);
  return output;
}
Tensor VariableType::normal(const Tensor & mean, double std, Generator * generator) const {
  profiler::RecordFunction profiler("normal");
  auto& mean_ = unpack(mean, "mean", 0);
  std::shared_ptr<NormalBackward1> grad_fn;
  if (compute_requires_grad( mean )) {
    grad_fn = std::make_shared<NormalBackward1>();
    grad_fn->set_next_edges(collect_next_edges( mean ));
    grad_fn->mean_sizes = mean.sizes();
  }
  auto output = as_variable(baseType->normal(mean_, std, generator));
  set_history(output, grad_fn);
  return output;
}
Tensor & VariableType::normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) const {
  profiler::RecordFunction profiler("normal_out");
  auto& output_ = unpack(output, "output", 0);
  auto& std_ = unpack(std, "std", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( std )) {
    throw_error_out_requires_grad("normal");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("normal");
  }
  baseType->normal_out(output_, mean, std_, generator);
  increment_version(output);
  rebase_history(output, grad_fn);
  return output;
}
Tensor VariableType::normal(double mean, const Tensor & std, Generator * generator) const {
  profiler::RecordFunction profiler("normal");
  auto& std_ = unpack(std, "std", 1);
  std::shared_ptr<NormalBackward2> grad_fn;
  if (compute_requires_grad( std )) {
    grad_fn = std::make_shared<NormalBackward2>();
    grad_fn->set_next_edges(collect_next_edges( std ));
    grad_fn->std_sizes = std.sizes();
  }
  auto output = as_variable(baseType->normal(mean, std_, generator));
  set_history(output, grad_fn);
  return output;
}
Tensor & VariableType::normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) const {
  profiler::RecordFunction profiler("normal_out");
  auto& output_ = unpack(output, "output", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  auto& std_ = unpack(std, "std", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( mean, std )) {
    throw_error_out_requires_grad("normal");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("normal");
  }
  baseType->normal_out(output_, mean_, std_, generator);
  increment_version(output);
  rebase_history(output, grad_fn);
  return output;
}
Tensor VariableType::normal(const Tensor & mean, const Tensor & std, Generator * generator) const {
  profiler::RecordFunction profiler("normal");
  auto& mean_ = unpack(mean, "mean", 0);
  auto& std_ = unpack(std, "std", 1);
  std::shared_ptr<NormalBackward3> grad_fn;
  if (compute_requires_grad( mean, std )) {
    grad_fn = std::make_shared<NormalBackward3>();
    grad_fn->set_next_edges(collect_next_edges( mean, std ));
    grad_fn->mean_sizes = mean.sizes();
    grad_fn->std_sizes = std.sizes();
  }
  auto output = as_variable(baseType->normal(mean_, std_, generator));
  set_history(output, grad_fn);
  return output;
}
Tensor & VariableType::normal_(Tensor & self, double mean, double std, Generator * generator) const {
  profiler::RecordFunction profiler("normal_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NormalBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<NormalBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  baseType->normal_(self_, mean, std, generator);
  increment_version(self);
  rebase_history(self, grad_fn);
  return self;
}
Tensor & VariableType::cauchy_(Tensor & self, double median, double sigma, Generator * generator) const {
  profiler::RecordFunction profiler("cauchy_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<CauchyBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<CauchyBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  baseType->cauchy_(self_, median, sigma, generator);
  increment_version(self);
  rebase_history(self, grad_fn);
  return self;
}
Tensor & VariableType::log_normal_(Tensor & self, double mean, double std, Generator * generator) const {
  profiler::RecordFunction profiler("log_normal_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LogNormalBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<LogNormalBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  baseType->log_normal_(self_, mean, std, generator);
  increment_version(self);
  rebase_history(self, grad_fn);
  return self;
}
Tensor & VariableType::exponential_(Tensor & self, double lambd, Generator * generator) const {
  profiler::RecordFunction profiler("exponential_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ExponentialBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ExponentialBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  baseType->exponential_(self_, lambd, generator);
  increment_version(self);
  rebase_history(self, grad_fn);
  return self;
}
Tensor & VariableType::rand_out(Tensor & result, IntList size, Generator * generator) const {
  profiler::RecordFunction profiler("rand_out");
  auto& result_ = unpack(result, "result", 0);
  baseType->rand_out(result_, size, generator);
  return result;
}
Tensor VariableType::rand(IntList size, Generator * generator) const {
  profiler::RecordFunction profiler("rand");
  auto result = as_variable(baseType->rand(size, generator));
  return result;
}
Tensor & VariableType::randn_out(Tensor & result, IntList size, Generator * generator) const {
  profiler::RecordFunction profiler("randn_out");
  auto& result_ = unpack(result, "result", 0);
  baseType->randn_out(result_, size, generator);
  return result;
}
Tensor VariableType::randn(IntList size, Generator * generator) const {
  profiler::RecordFunction profiler("randn");
  auto result = as_variable(baseType->randn(size, generator));
  return result;
}
Tensor & VariableType::geometric_(Tensor & self, double p, Generator * generator) const {
  profiler::RecordFunction profiler("geometric_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<GeometricBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<GeometricBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  baseType->geometric_(self_, p, generator);
  increment_version(self);
  rebase_history(self, grad_fn);
  return self;
}
Tensor & VariableType::bernoulli_out(Tensor & output, const Tensor & self, Generator * generator) const {
  profiler::RecordFunction profiler("bernoulli_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("bernoulli");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("bernoulli");
  }
  baseType->bernoulli_out(output_, self_, generator);
  increment_version(output);
  rebase_history(output, grad_fn);
  return output;
}
Tensor VariableType::bernoulli(const Tensor & self, Generator * generator) const {
  profiler::RecordFunction profiler("bernoulli");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<BernoulliBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<BernoulliBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  auto output = as_variable(baseType->bernoulli(self_, generator));
  set_history(output, grad_fn);
  return output;
}
Tensor & VariableType::_standard_gamma_out(Tensor & output, const Tensor & self, Generator * generator) const {
  profiler::RecordFunction profiler("_standard_gamma_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_standard_gamma");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_standard_gamma");
  }
  baseType->_standard_gamma_out(output_, self_, generator);
  increment_version(output);
  rebase_history(output, grad_fn);
  return output;
}
Tensor VariableType::_standard_gamma(const Tensor & self, Generator * generator) const {
  profiler::RecordFunction profiler("_standard_gamma");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<StandardGammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<StandardGammaBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  auto output = as_variable(baseType->_standard_gamma(self_, generator));
  set_history(output, grad_fn);
  if (grad_fn) {
    grad_fn->output_ = SavedVariable(output, true);
  }
  return output;
}
Tensor & VariableType::_dirichlet_grad_out(Tensor & output, const Tensor & x, const Tensor & alpha, const Tensor & total) const {
  profiler::RecordFunction profiler("_dirichlet_grad_out");
  auto& output_ = unpack(output, "output", 0);
  auto& x_ = unpack(x, "x", 1);
  auto& alpha_ = unpack(alpha, "alpha", 2);
  auto& total_ = unpack(total, "total", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( x, alpha, total )) {
    throw_error_out_requires_grad("_dirichlet_grad");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_dirichlet_grad");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, x, alpha, total )) {
    trace_info = jit::tracer::preRecordTrace( "_dirichlet_grad_out", { output, x, alpha, total } );
  
  }
  baseType->_dirichlet_grad_out(output_, x_, alpha_, total_);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::_dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total) const {
  profiler::RecordFunction profiler("_dirichlet_grad");
  auto& x_ = unpack(x, "x", 0);
  auto& alpha_ = unpack(alpha, "alpha", 1);
  auto& total_ = unpack(total, "total", 2);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( x, alpha, total )) {
    grad_fn = std::make_shared<Error>("the derivative for _dirichlet_grad is not implemented");
    grad_fn->set_next_edges(collect_next_edges( x, alpha, total ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( x, alpha, total )) {
    trace_info = jit::tracer::preRecordTrace( "_dirichlet_grad", { x, alpha, total } );
  
  }
  auto output = as_variable(baseType->_dirichlet_grad(x_, alpha_, total_));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor VariableType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
  profiler::RecordFunction profiler("tensor");
  auto result = as_variable(baseType->tensor(storage, storageOffset, size, stride));
  return result;
}
Tensor VariableType::tensor(IntList size) const {
  profiler::RecordFunction profiler("tensor");
  auto result = as_variable(baseType->tensor(size));
  return result;
}
Tensor VariableType::tensor(IntList size, IntList stride) const {
  profiler::RecordFunction profiler("tensor");
  auto result = as_variable(baseType->tensor(size, stride));
  return result;
}
Tensor VariableType::tensor() const {
  profiler::RecordFunction profiler("tensor");
  auto result = as_variable(baseType->tensor());
  return result;
}
Tensor VariableType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) const {
  profiler::RecordFunction profiler("sparse_coo_tensor");
  auto& indices_ = unpack(indices, "indices", 0);
  auto& values_ = unpack(values, "values", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( indices, values )) {
    trace_info = jit::tracer::preRecordTrace( "sparse_coo_tensor", { indices, values } );
    setattr(trace_info.n, jit::Symbol("size"), size);
  }
  auto result = as_variable(baseType->sparse_coo_tensor(indices_, values_, size));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::sparse_coo_tensor(const Tensor & indices, const Tensor & values) const {
  profiler::RecordFunction profiler("sparse_coo_tensor");
  auto& indices_ = unpack(indices, "indices", 0);
  auto& values_ = unpack(values, "values", 1);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( indices, values )) {
    trace_info = jit::tracer::preRecordTrace( "sparse_coo_tensor", { indices, values } );
  
  }
  auto result = as_variable(baseType->sparse_coo_tensor(indices_, values_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::alias(const Tensor & self) const {
  profiler::RecordFunction profiler("alias");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AliasBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AliasBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "alias", { self } );
  
  }
  auto result = as_view(self, baseType->alias(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::_copy_ignoring_overlaps_(Tensor & self, const Tensor & src) const {
  profiler::RecordFunction profiler("_copy_ignoring_overlaps_");
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::make_shared<Error>("the derivative for _copy_ignoring_overlaps_ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, src ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, src )) {
    trace_info = jit::tracer::preRecordTrace( "_copy_ignoring_overlaps", { self, src } );
  
  }
  baseType->_copy_ignoring_overlaps_(self_, src_);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
  profiler::RecordFunction profiler("as_strided_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("as_strided");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("as_strided");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self )) {
    trace_info = jit::tracer::preRecordTrace( "as_strided_out", { result, self } );
    setattr(trace_info.n, jit::Symbol("size"), size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("storage_offset"), storage_offset);
  }
  baseType->as_strided_out(result_, self_, size, stride, storage_offset);
  increment_version(result);
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
  profiler::RecordFunction profiler("as_strided");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AsStridedBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AsStridedBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_geometry = TensorGeometry(self);
    grad_fn->size = size;
    grad_fn->stride = stride;
    grad_fn->storage_offset = storage_offset;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "as_strided", { self } );
    setattr(trace_info.n, jit::Symbol("size"), size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("storage_offset"), storage_offset);
  }
  auto result = as_view(self, baseType->as_strided(self_, size, stride, storage_offset));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
  profiler::RecordFunction profiler("as_strided_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AsStridedBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AsStridedBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_geometry = TensorGeometry(self);
    grad_fn->size = size;
    grad_fn->stride = stride;
    grad_fn->storage_offset = storage_offset;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "as_strided", { self } );
    setattr(trace_info.n, jit::Symbol("size"), size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("storage_offset"), storage_offset);
  }
  baseType->as_strided_(self_, size, stride, storage_offset);
  increment_version(self);
  set_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::sparse_raw_resize_(Tensor & self, IntList size, int64_t nDimI, int64_t nDimV) const {
  profiler::RecordFunction profiler("sparse_raw_resize_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for sparse_raw_resize_ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "sparse_raw_resize", { self } );
    setattr(trace_info.n, jit::Symbol("size"), size);
    setattr(trace_info.n, jit::Symbol("nDimI"), nDimI);
    setattr(trace_info.n, jit::Symbol("nDimV"), nDimV);
  }
  baseType->sparse_raw_resize_(self_, size, nDimI, nDimV);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
  profiler::RecordFunction profiler("_cat_out");
  auto& self_ = unpack(self, "self", 0);
  auto tensors_ = unpack(tensors, "tensors", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( tensors )) {
    throw_error_out_requires_grad("_cat");
  }
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_cat");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, tensors )) {
    trace_info = jit::tracer::preRecordTrace( "_cat_out", flatten( self, tensors ) );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->_cat_out(self_, tensors_, dim);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {self} );
  }
  return self;
}
Tensor VariableType::_cat(TensorList tensors, int64_t dim) const {
  profiler::RecordFunction profiler("_cat");
  auto tensors_ = unpack(tensors, "tensors", 0);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( tensors )) {
    grad_fn = std::make_shared<Error>("the derivative for _cat is not implemented");
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( tensors )) {
    trace_info = jit::tracer::preRecordTrace( "_cat", flatten( tensors ) );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto self = as_variable(baseType->_cat(tensors_, dim));
  set_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::reshape_(Tensor & self, IntList size, IntList stride) const {
  profiler::RecordFunction profiler("reshape_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for reshape_ is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "reshape", { self } );
    setattr(trace_info.n, jit::Symbol("size"), size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
  }
  baseType->reshape_(self_, size, stride);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor VariableType::_sparse_mask(const Tensor & self, SparseTensor mask) const {
  profiler::RecordFunction profiler("_sparse_mask");
  auto& self_ = unpack(self, "self", 0);
  auto mask_ = unpack(mask, "mask", 1);
  std::shared_ptr<SparseMaskBackward> grad_fn;
  if (compute_requires_grad( self, mask.tref )) {
    grad_fn = std::make_shared<SparseMaskBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, mask.tref ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "_sparse_mask", { self } );
    setattr(trace_info.n, jit::Symbol("mask"), mask);
  }
  auto result = as_variable(baseType->_sparse_mask(self_, mask_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::to_dense(const Tensor & self) const {
  profiler::RecordFunction profiler("to_dense");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for to_dense is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "to_dense", { self } );
  
  }
  auto result = as_variable(baseType->to_dense(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
int64_t VariableType::_dimI(const Tensor & self) const {
  profiler::RecordFunction profiler("_dimI");
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->_dimI(self_);
  return result;
}
int64_t VariableType::_dimV(const Tensor & self) const {
  profiler::RecordFunction profiler("_dimV");
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->_dimV(self_);
  return result;
}
int64_t VariableType::_nnz(const Tensor & self) const {
  profiler::RecordFunction profiler("_nnz");
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->_nnz(self_);
  return result;
}
Tensor VariableType::coalesce(const Tensor & self) const {
  profiler::RecordFunction profiler("coalesce");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for coalesce is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "coalesce", { self } );
  
  }
  auto result = as_variable(baseType->coalesce(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
bool VariableType::is_coalesced(const Tensor & self) const {
  profiler::RecordFunction profiler("is_coalesced");
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->is_coalesced(self_);
  return result;
}
Tensor VariableType::_indices(const Tensor & self) const {
  profiler::RecordFunction profiler("_indices");
  auto& self_ = unpack(self, "self", 0);
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "_indices", { self } );
  
  }
  auto result = as_variable(baseType->_indices(self_));
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::_values(const Tensor & self) const {
  profiler::RecordFunction profiler("_values");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for _values is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "_values", { self } );
  
  }
  auto result = as_variable(baseType->_values(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::hspmm_out(Tensor & result, const Tensor & mat1, const Tensor & mat2) const {
  profiler::RecordFunction profiler("hspmm_out");
  auto& result_ = unpack(result, "result", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( mat1, mat2 )) {
    throw_error_out_requires_grad("hspmm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("hspmm");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, mat1, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "hspmm_out", { result, mat1, mat2 } );
  
  }
  baseType->hspmm_out(result_, mat1_, mat2_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
  profiler::RecordFunction profiler("hspmm");
  auto& mat1_ = unpack(mat1, "mat1", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( mat1, mat2 )) {
    grad_fn = std::make_shared<Error>("the derivative for hspmm is not implemented");
    grad_fn->set_next_edges(collect_next_edges( mat1, mat2 ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( mat1, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "hspmm", { mat1, mat2 } );
  
  }
  auto result = as_variable(baseType->hspmm(mat1_, mat2_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::binary_cross_entropy_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("binary_cross_entropy_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "binary_cross_entropy_out", { output, self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  Type::binary_cross_entropy_out(output, self, target, weight, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("binary_cross_entropy");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "binary_cross_entropy", { self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = Type::binary_cross_entropy(self, target, weight, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::binary_cross_entropy_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("binary_cross_entropy_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target, weight )) {
    throw_error_out_requires_grad("binary_cross_entropy_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("binary_cross_entropy_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "binary_cross_entropy_forward_out", { output, self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->binary_cross_entropy_forward_out(output_, self_, target_, weight_, size_average, reduce);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::binary_cross_entropy_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("binary_cross_entropy_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 2);
  check_no_requires_grad(target, "target");
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<BinaryCrossEntropyBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<BinaryCrossEntropyBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->size_average = size_average;
    grad_fn->reduce = reduce;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "binary_cross_entropy_forward", { self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = as_variable(baseType->binary_cross_entropy_forward(self_, target_, weight_, size_average, reduce));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("binary_cross_entropy_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target, weight )) {
    throw_error_out_requires_grad("binary_cross_entropy_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("binary_cross_entropy_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "binary_cross_entropy_backward_out", { grad_input, grad_output, self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->binary_cross_entropy_backward_out(grad_input_, grad_output_, self_, target_, weight_, size_average, reduce);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("binary_cross_entropy_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 3);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( grad_output, self, target, weight )) {
    grad_fn = std::make_shared<Error>("the derivative for binary_cross_entropy_backward is not implemented");
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target, weight ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "binary_cross_entropy_backward", { grad_output, self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto grad_input = as_variable(baseType->binary_cross_entropy_backward(grad_output_, self_, target_, weight_, size_average, reduce));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::kl_div_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("kl_div_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "kl_div_out", { output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  Type::kl_div_out(output, self, target, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::kl_div(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("kl_div");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target )) {
    trace_info = jit::tracer::preRecordTrace( "kl_div", { self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = Type::kl_div(self, target, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::kl_div_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("kl_div_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("kl_div_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("kl_div_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "kl_div_forward_out", { output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->kl_div_forward_out(output_, self_, target_, size_average, reduce);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::kl_div_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("kl_div_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  check_no_requires_grad(target, "target");
  std::shared_ptr<KlDivBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<KlDivBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->size_average = size_average;
    grad_fn->reduce = reduce;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target )) {
    trace_info = jit::tracer::preRecordTrace( "kl_div_forward", { self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = as_variable(baseType->kl_div_forward(self_, target_, size_average, reduce));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::kl_div_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("kl_div_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("kl_div_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("kl_div_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "kl_div_backward_out", { grad_input, grad_output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->kl_div_backward_out(grad_input_, grad_output_, self_, target_, size_average, reduce);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("kl_div_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<KlDivBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<KlDivBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->size_average = size_average;
    grad_fn->reduce = reduce;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "kl_div_backward", { grad_output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto grad_input = as_variable(baseType->kl_div_backward(grad_output_, self_, target_, size_average, reduce));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("l1_loss_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "l1_loss_out", { output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  Type::l1_loss_out(output, self, target, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::l1_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("l1_loss");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target )) {
    trace_info = jit::tracer::preRecordTrace( "l1_loss", { self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = Type::l1_loss(self, target, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("l1_loss_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("l1_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("l1_loss_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "l1_loss_forward_out", { output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->l1_loss_forward_out(output_, self_, target_, size_average, reduce);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("l1_loss_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  check_no_requires_grad(target, "target");
  std::shared_ptr<L1LossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<L1LossBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->size_average = size_average;
    grad_fn->reduce = reduce;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target )) {
    trace_info = jit::tracer::preRecordTrace( "l1_loss_forward", { self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = as_variable(baseType->l1_loss_forward(self_, target_, size_average, reduce));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("l1_loss_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("l1_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("l1_loss_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "l1_loss_backward_out", { grad_input, grad_output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->l1_loss_backward_out(grad_input_, grad_output_, self_, target_, size_average, reduce);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("l1_loss_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<L1LossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<L1LossBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->size_average = size_average;
    grad_fn->reduce = reduce;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "l1_loss_backward", { grad_output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto grad_input = as_variable(baseType->l1_loss_backward(grad_output_, self_, target_, size_average, reduce));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::mse_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("mse_loss_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "mse_loss_out", { output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  Type::mse_loss_out(output, self, target, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::mse_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("mse_loss");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target )) {
    trace_info = jit::tracer::preRecordTrace( "mse_loss", { self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = Type::mse_loss(self, target, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::mse_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("mse_loss_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("mse_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("mse_loss_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "mse_loss_forward_out", { output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->mse_loss_forward_out(output_, self_, target_, size_average, reduce);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::mse_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("mse_loss_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  check_no_requires_grad(target, "target");
  std::shared_ptr<MseLossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MseLossBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->size_average = size_average;
    grad_fn->reduce = reduce;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target )) {
    trace_info = jit::tracer::preRecordTrace( "mse_loss_forward", { self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = as_variable(baseType->mse_loss_forward(self_, target_, size_average, reduce));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("mse_loss_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("mse_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("mse_loss_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "mse_loss_backward_out", { grad_input, grad_output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->mse_loss_backward_out(grad_input_, grad_output_, self_, target_, size_average, reduce);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("mse_loss_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<MseLossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<MseLossBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->size_average = size_average;
    grad_fn->reduce = reduce;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "mse_loss_backward", { grad_output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto grad_input = as_variable(baseType->mse_loss_backward(grad_output_, self_, target_, size_average, reduce));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::multi_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const {
  profiler::RecordFunction profiler("multi_margin_loss_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "multi_margin_loss_out", { output, self, target, weight } );
    setattr(trace_info.n, jit::Symbol("p"), p);
    setattr(trace_info.n, jit::Symbol("margin"), margin);
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
  }
  Type::multi_margin_loss_out(output, self, target, p, margin, weight, size_average);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const {
  profiler::RecordFunction profiler("multi_margin_loss");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "multi_margin_loss", { self, target, weight } );
    setattr(trace_info.n, jit::Symbol("p"), p);
    setattr(trace_info.n, jit::Symbol("margin"), margin);
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
  }
  auto output = Type::multi_margin_loss(self, target, p, margin, weight, size_average);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const {
  profiler::RecordFunction profiler("multi_margin_loss_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight )) {
    throw_error_out_requires_grad("multi_margin_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("multi_margin_loss_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "multi_margin_loss_forward_out", { output, self, target, weight } );
    setattr(trace_info.n, jit::Symbol("p"), p);
    setattr(trace_info.n, jit::Symbol("margin"), margin);
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
  }
  baseType->multi_margin_loss_forward_out(output_, self_, target_, p, margin, weight_, size_average);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::multi_margin_loss_forward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const {
  profiler::RecordFunction profiler("multi_margin_loss_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 4);
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<MultiMarginLossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MultiMarginLossBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->p = p;
    grad_fn->margin = margin;
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->size_average = size_average;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "multi_margin_loss_forward", { self, target, weight } );
    setattr(trace_info.n, jit::Symbol("p"), p);
    setattr(trace_info.n, jit::Symbol("margin"), margin);
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
  }
  auto output = as_variable(baseType->multi_margin_loss_forward(self_, target_, p, margin, weight_, size_average));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const {
  profiler::RecordFunction profiler("multi_margin_loss_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight )) {
    throw_error_out_requires_grad("multi_margin_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("multi_margin_loss_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "multi_margin_loss_backward_out", { grad_input, self, target, weight } );
    setattr(trace_info.n, jit::Symbol("p"), p);
    setattr(trace_info.n, jit::Symbol("margin"), margin);
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
  }
  baseType->multi_margin_loss_backward_out(grad_input_, self_, target_, p, margin, weight_, size_average);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::multi_margin_loss_backward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const {
  profiler::RecordFunction profiler("multi_margin_loss_backward");
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 4);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self, weight )) {
    grad_fn = std::make_shared<Error>("the derivative for multi_margin_loss_backward is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self, weight ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "multi_margin_loss_backward", { self, target, weight } );
    setattr(trace_info.n, jit::Symbol("p"), p);
    setattr(trace_info.n, jit::Symbol("margin"), margin);
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
  }
  auto grad_input = as_variable(baseType->multi_margin_loss_backward(self_, target_, p, margin, weight_, size_average));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::multilabel_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("multilabel_margin_loss_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "multilabel_margin_loss_out", { output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  Type::multilabel_margin_loss_out(output, self, target, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::multilabel_margin_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("multilabel_margin_loss");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target )) {
    trace_info = jit::tracer::preRecordTrace( "multilabel_margin_loss", { self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = Type::multilabel_margin_loss(self, target, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &> VariableType::multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("multilabel_margin_loss_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& is_target_ = unpack(is_target, "is_target", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("multilabel_margin_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("multilabel_margin_loss_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, is_target, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "multilabel_margin_loss_forward_out", { output, is_target, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->multilabel_margin_loss_forward_out(output_, is_target_, self_, target_, size_average, reduce);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, is_target} );
  }
  return std::forward_as_tuple(output, is_target);
}
std::tuple<Tensor,Tensor> VariableType::multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("multilabel_margin_loss_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  std::shared_ptr<MultilabelMarginLossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MultilabelMarginLossBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->size_average = size_average;
    grad_fn->reduce = reduce;
  }
  Tensor output;
  Tensor is_target;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target )) {
    trace_info = jit::tracer::preRecordTrace( "multilabel_margin_loss_forward", { self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  std::tie(output, is_target) = as_variable(baseType->multilabel_margin_loss_forward(self_, target_, size_average, reduce));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, is_target } );
  }
  if (grad_fn) {
    grad_fn->is_target_ = SavedVariable(is_target, true);
  }
  return std::make_tuple(std::move(output), std::move(is_target));
}
Tensor & VariableType::multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
  profiler::RecordFunction profiler("multilabel_margin_loss_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto& is_target_ = unpack(is_target, "is_target", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, is_target )) {
    throw_error_out_requires_grad("multilabel_margin_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("multilabel_margin_loss_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, target, is_target )) {
    trace_info = jit::tracer::preRecordTrace( "multilabel_margin_loss_backward_out", { grad_input, grad_output, self, target, is_target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->multilabel_margin_loss_backward_out(grad_input_, grad_output_, self_, target_, size_average, reduce, is_target_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
  profiler::RecordFunction profiler("multilabel_margin_loss_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& is_target_ = unpack(is_target, "is_target", 5);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( grad_output, self, is_target )) {
    grad_fn = std::make_shared<Error>("the derivative for multilabel_margin_loss_backward is not implemented");
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, is_target ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, target, is_target )) {
    trace_info = jit::tracer::preRecordTrace( "multilabel_margin_loss_backward", { grad_output, self, target, is_target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto grad_input = as_variable(baseType->multilabel_margin_loss_backward(grad_output_, self_, target_, size_average, reduce, is_target_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::nll_loss_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
  profiler::RecordFunction profiler("nll_loss_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "nll_loss_out", { output, self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("ignore_index"), ignore_index);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  Type::nll_loss_out(output, self, target, weight, size_average, ignore_index, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
  profiler::RecordFunction profiler("nll_loss");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "nll_loss", { self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("ignore_index"), ignore_index);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = Type::nll_loss(self, target, weight, size_average, ignore_index, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &> VariableType::nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
  profiler::RecordFunction profiler("nll_loss_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& total_weight_ = unpack(total_weight, "total_weight", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight )) {
    throw_error_out_requires_grad("nll_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("nll_loss_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, total_weight, self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "nll_loss_forward_out", { output, total_weight, self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("ignore_index"), ignore_index);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->nll_loss_forward_out(output_, total_weight_, self_, target_, weight_, size_average, ignore_index, reduce);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, total_weight} );
  }
  return std::forward_as_tuple(output, total_weight);
}
std::tuple<Tensor,Tensor> VariableType::nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
  profiler::RecordFunction profiler("nll_loss_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 2);
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<NllLossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<NllLossBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->size_average = size_average;
    grad_fn->ignore_index = ignore_index;
    grad_fn->reduce = reduce;
  }
  Tensor output;
  Tensor total_weight;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "nll_loss_forward", { self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("ignore_index"), ignore_index);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  std::tie(output, total_weight) = as_variable(baseType->nll_loss_forward(self_, target_, weight_, size_average, ignore_index, reduce));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, total_weight } );
  }
  if (grad_fn) {
    grad_fn->total_weight_ = SavedVariable(total_weight, true);
  }
  return std::make_tuple(std::move(output), std::move(total_weight));
}
Tensor & VariableType::nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
  profiler::RecordFunction profiler("nll_loss_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  auto& total_weight_ = unpack(total_weight, "total_weight", 8);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, total_weight )) {
    throw_error_out_requires_grad("nll_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("nll_loss_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, target, weight, total_weight )) {
    trace_info = jit::tracer::preRecordTrace( "nll_loss_backward_out", { grad_input, grad_output, self, target, weight, total_weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("ignore_index"), ignore_index);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->nll_loss_backward_out(grad_input_, grad_output_, self_, target_, weight_, size_average, ignore_index, reduce, total_weight_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
  profiler::RecordFunction profiler("nll_loss_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 3);
  auto& total_weight_ = unpack(total_weight, "total_weight", 7);
  check_no_requires_grad(weight, "weight");
  check_no_requires_grad(total_weight, "total_weight");
  std::shared_ptr<NllLossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<NllLossBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->size_average = size_average;
    grad_fn->ignore_index = ignore_index;
    grad_fn->reduce = reduce;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, target, weight, total_weight )) {
    trace_info = jit::tracer::preRecordTrace( "nll_loss_backward", { grad_output, self, target, weight, total_weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("ignore_index"), ignore_index);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto grad_input = as_variable(baseType->nll_loss_backward(grad_output_, self_, target_, weight_, size_average, ignore_index, reduce, total_weight_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::nll_loss2d_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
  profiler::RecordFunction profiler("nll_loss2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "nll_loss2d_out", { output, self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("ignore_index"), ignore_index);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  Type::nll_loss2d_out(output, self, target, weight, size_average, ignore_index, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
  profiler::RecordFunction profiler("nll_loss2d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "nll_loss2d", { self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("ignore_index"), ignore_index);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = Type::nll_loss2d(self, target, weight, size_average, ignore_index, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &> VariableType::nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
  profiler::RecordFunction profiler("nll_loss2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& total_weight_ = unpack(total_weight, "total_weight", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight )) {
    throw_error_out_requires_grad("nll_loss2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("nll_loss2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, total_weight, self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "nll_loss2d_forward_out", { output, total_weight, self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("ignore_index"), ignore_index);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->nll_loss2d_forward_out(output_, total_weight_, self_, target_, weight_, size_average, ignore_index, reduce);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, total_weight} );
  }
  return std::forward_as_tuple(output, total_weight);
}
std::tuple<Tensor,Tensor> VariableType::nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
  profiler::RecordFunction profiler("nll_loss2d_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 2);
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<NllLoss2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<NllLoss2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->size_average = size_average;
    grad_fn->ignore_index = ignore_index;
    grad_fn->reduce = reduce;
  }
  Tensor output;
  Tensor total_weight;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target, weight )) {
    trace_info = jit::tracer::preRecordTrace( "nll_loss2d_forward", { self, target, weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("ignore_index"), ignore_index);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  std::tie(output, total_weight) = as_variable(baseType->nll_loss2d_forward(self_, target_, weight_, size_average, ignore_index, reduce));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, total_weight } );
  }
  if (grad_fn) {
    grad_fn->total_weight_ = SavedVariable(total_weight, true);
  }
  return std::make_tuple(std::move(output), std::move(total_weight));
}
Tensor & VariableType::nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
  profiler::RecordFunction profiler("nll_loss2d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  auto& total_weight_ = unpack(total_weight, "total_weight", 8);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, total_weight )) {
    throw_error_out_requires_grad("nll_loss2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("nll_loss2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, target, weight, total_weight )) {
    trace_info = jit::tracer::preRecordTrace( "nll_loss2d_backward_out", { grad_input, grad_output, self, target, weight, total_weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("ignore_index"), ignore_index);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->nll_loss2d_backward_out(grad_input_, grad_output_, self_, target_, weight_, size_average, ignore_index, reduce, total_weight_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
  profiler::RecordFunction profiler("nll_loss2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 3);
  auto& total_weight_ = unpack(total_weight, "total_weight", 7);
  check_no_requires_grad(weight, "weight");
  check_no_requires_grad(total_weight, "total_weight");
  std::shared_ptr<NllLoss2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<NllLoss2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->size_average = size_average;
    grad_fn->ignore_index = ignore_index;
    grad_fn->reduce = reduce;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, target, weight, total_weight )) {
    trace_info = jit::tracer::preRecordTrace( "nll_loss2d_backward", { grad_output, self, target, weight, total_weight } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("ignore_index"), ignore_index);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto grad_input = as_variable(baseType->nll_loss2d_backward(grad_output_, self_, target_, weight_, size_average, ignore_index, reduce, total_weight_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::smooth_l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("smooth_l1_loss_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "smooth_l1_loss_out", { output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  Type::smooth_l1_loss_out(output, self, target, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::smooth_l1_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("smooth_l1_loss");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target )) {
    trace_info = jit::tracer::preRecordTrace( "smooth_l1_loss", { self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = Type::smooth_l1_loss(self, target, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("smooth_l1_loss_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("smooth_l1_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("smooth_l1_loss_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "smooth_l1_loss_forward_out", { output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->smooth_l1_loss_forward_out(output_, self_, target_, size_average, reduce);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::smooth_l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("smooth_l1_loss_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  check_no_requires_grad(target, "target");
  std::shared_ptr<SmoothL1LossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SmoothL1LossBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->size_average = size_average;
    grad_fn->reduce = reduce;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target )) {
    trace_info = jit::tracer::preRecordTrace( "smooth_l1_loss_forward", { self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = as_variable(baseType->smooth_l1_loss_forward(self_, target_, size_average, reduce));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("smooth_l1_loss_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("smooth_l1_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("smooth_l1_loss_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "smooth_l1_loss_backward_out", { grad_input, grad_output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->smooth_l1_loss_backward_out(grad_input_, grad_output_, self_, target_, size_average, reduce);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("smooth_l1_loss_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<SmoothL1LossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<SmoothL1LossBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->size_average = size_average;
    grad_fn->reduce = reduce;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "smooth_l1_loss_backward", { grad_output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto grad_input = as_variable(baseType->smooth_l1_loss_backward(grad_output_, self_, target_, size_average, reduce));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::soft_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("soft_margin_loss_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "soft_margin_loss_out", { output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  Type::soft_margin_loss_out(output, self, target, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::soft_margin_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("soft_margin_loss");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target )) {
    trace_info = jit::tracer::preRecordTrace( "soft_margin_loss", { self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = Type::soft_margin_loss(self, target, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::soft_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("soft_margin_loss_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("soft_margin_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("soft_margin_loss_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "soft_margin_loss_forward_out", { output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->soft_margin_loss_forward_out(output_, self_, target_, size_average, reduce);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::soft_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("soft_margin_loss_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  check_no_requires_grad(target, "target");
  std::shared_ptr<SoftMarginLossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SoftMarginLossBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->size_average = size_average;
    grad_fn->reduce = reduce;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target )) {
    trace_info = jit::tracer::preRecordTrace( "soft_margin_loss_forward", { self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto output = as_variable(baseType->soft_margin_loss_forward(self_, target_, size_average, reduce));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("soft_margin_loss_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("soft_margin_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("soft_margin_loss_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "soft_margin_loss_backward_out", { grad_input, grad_output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  baseType->soft_margin_loss_backward_out(grad_input_, grad_output_, self_, target_, size_average, reduce);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("soft_margin_loss_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<SoftMarginLossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<SoftMarginLossBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->size_average = size_average;
    grad_fn->reduce = reduce;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, target )) {
    trace_info = jit::tracer::preRecordTrace( "soft_margin_loss_backward", { grad_output, self, target } );
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto grad_input = as_variable(baseType->soft_margin_loss_backward(grad_output_, self_, target_, size_average, reduce));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::elu_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale) const {
  profiler::RecordFunction profiler("elu_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "elu_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
    setattr(trace_info.n, jit::Symbol("scale"), scale);
  }
  Type::elu_out(output, self, alpha, scale);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::elu(const Tensor & self, Scalar alpha, Scalar scale) const {
  profiler::RecordFunction profiler("elu");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "elu", { self } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
    setattr(trace_info.n, jit::Symbol("scale"), scale);
  }
  auto output = Type::elu(self, alpha, scale);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::elu_forward_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale) const {
  profiler::RecordFunction profiler("elu_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("elu_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("elu_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "elu_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
    setattr(trace_info.n, jit::Symbol("scale"), scale);
  }
  baseType->elu_forward_out(output_, self_, alpha, scale);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::elu_forward(const Tensor & self, Scalar alpha, Scalar scale) const {
  profiler::RecordFunction profiler("elu_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<EluBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<EluBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->alpha = alpha;
    grad_fn->scale = scale;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "elu_forward", { self } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
    setattr(trace_info.n, jit::Symbol("scale"), scale);
  }
  auto output = as_variable(baseType->elu_forward(self_, alpha, scale));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  if (grad_fn) {
    grad_fn->output_ = SavedVariable(output, true);
  }
  return output;
}
Tensor & VariableType::elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
  profiler::RecordFunction profiler("elu_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& output_ = unpack(output, "output", 4);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    throw_error_out_requires_grad("elu_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("elu_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, output )) {
    trace_info = jit::tracer::preRecordTrace( "elu_backward_out", { grad_input, grad_output, output } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
    setattr(trace_info.n, jit::Symbol("scale"), scale);
  }
  baseType->elu_backward_out(grad_input_, grad_output_, alpha, scale, output_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
  profiler::RecordFunction profiler("elu_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 3);
  std::shared_ptr<EluBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    grad_fn = std::make_shared<EluBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, output ));
    grad_fn->alpha = alpha;
    grad_fn->scale = scale;
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, output )) {
    trace_info = jit::tracer::preRecordTrace( "elu_backward", { grad_output, output } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
    setattr(trace_info.n, jit::Symbol("scale"), scale);
  }
  auto grad_input = as_variable(baseType->elu_backward(grad_output_, alpha, scale, output_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::elu_(Tensor & self, Scalar alpha, Scalar scale) const {
  profiler::RecordFunction profiler("elu_");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "elu", { self } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
    setattr(trace_info.n, jit::Symbol("scale"), scale);
  }
  Type::elu_(self, alpha, scale);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::elu_forward_(Tensor & self, Scalar alpha, Scalar scale) const {
  profiler::RecordFunction profiler("elu_forward_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<EluBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<EluBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->alpha = alpha;
    grad_fn->scale = scale;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "elu_forward", { self } );
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
    setattr(trace_info.n, jit::Symbol("scale"), scale);
  }
  baseType->elu_forward_(self_, alpha, scale);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  if (grad_fn) {
    grad_fn->output_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::glu_out(Tensor & output, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("glu_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "glu_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  Type::glu_out(output, self, dim);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::glu(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("glu");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "glu", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto output = Type::glu(self, dim);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::glu_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("glu_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("glu_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("glu_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "glu_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->glu_forward_out(output_, self_, dim);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::glu_forward(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("glu_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<GluBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<GluBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "glu_forward", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto output = as_variable(baseType->glu_forward(self_, dim));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("glu_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("glu_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("glu_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "glu_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->glu_backward_out(grad_input_, grad_output_, self_, dim);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("glu_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<GluBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<GluBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "glu_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto grad_input = as_variable(baseType->glu_backward(grad_output_, self_, dim));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::hardshrink_out(Tensor & output, const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("hardshrink_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "hardshrink_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("lambd"), lambd);
  }
  Type::hardshrink_out(output, self, lambd);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::hardshrink(const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("hardshrink");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "hardshrink", { self } );
    setattr(trace_info.n, jit::Symbol("lambd"), lambd);
  }
  auto output = Type::hardshrink(self, lambd);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::hardshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("hardshrink_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("hardshrink_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("hardshrink_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "hardshrink_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("lambd"), lambd);
  }
  baseType->hardshrink_forward_out(output_, self_, lambd);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::hardshrink_forward(const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("hardshrink_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<HardshrinkBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<HardshrinkBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->lambd = lambd;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "hardshrink_forward", { self } );
    setattr(trace_info.n, jit::Symbol("lambd"), lambd);
  }
  auto output = as_variable(baseType->hardshrink_forward(self_, lambd));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::hardshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("hardshrink_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("hardshrink_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("hardshrink_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "hardshrink_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("lambd"), lambd);
  }
  baseType->hardshrink_backward_out(grad_input_, grad_output_, self_, lambd);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::hardshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("hardshrink_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<HardshrinkBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<HardshrinkBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->lambd = lambd;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "hardshrink_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("lambd"), lambd);
  }
  auto grad_input = as_variable(baseType->hardshrink_backward(grad_output_, self_, lambd));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::hardtanh_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "hardtanh_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("min_val"), min_val);
    setattr(trace_info.n, jit::Symbol("max_val"), max_val);
  }
  Type::hardtanh_out(output, self, min_val, max_val);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::hardtanh(const Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "hardtanh", { self } );
    setattr(trace_info.n, jit::Symbol("min_val"), min_val);
    setattr(trace_info.n, jit::Symbol("max_val"), max_val);
  }
  auto output = Type::hardtanh(self, min_val, max_val);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("hardtanh_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("hardtanh_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "hardtanh_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("min_val"), min_val);
    setattr(trace_info.n, jit::Symbol("max_val"), max_val);
  }
  baseType->hardtanh_forward_out(output_, self_, min_val, max_val);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::hardtanh_forward(const Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<HardtanhBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<HardtanhBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "hardtanh_forward", { self } );
    setattr(trace_info.n, jit::Symbol("min_val"), min_val);
    setattr(trace_info.n, jit::Symbol("max_val"), max_val);
  }
  auto output = as_variable(baseType->hardtanh_forward(self_, min_val, max_val));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("hardtanh_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("hardtanh_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "hardtanh_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("min_val"), min_val);
    setattr(trace_info.n, jit::Symbol("max_val"), max_val);
  }
  baseType->hardtanh_backward_out(grad_input_, grad_output_, self_, min_val, max_val);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<HardtanhBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<HardtanhBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "hardtanh_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("min_val"), min_val);
    setattr(trace_info.n, jit::Symbol("max_val"), max_val);
  }
  auto grad_input = as_variable(baseType->hardtanh_backward(grad_output_, self_, min_val, max_val));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::hardtanh_(Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh_");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "hardtanh", { self } );
    setattr(trace_info.n, jit::Symbol("min_val"), min_val);
    setattr(trace_info.n, jit::Symbol("max_val"), max_val);
  }
  Type::hardtanh_(self, min_val, max_val);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::hardtanh_forward_(Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh_forward_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<HardtanhBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<HardtanhBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "hardtanh_forward", { self } );
    setattr(trace_info.n, jit::Symbol("min_val"), min_val);
    setattr(trace_info.n, jit::Symbol("max_val"), max_val);
  }
  baseType->hardtanh_forward_(self_, min_val, max_val);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  if (grad_fn) {
    grad_fn->output_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::leaky_relu_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("leaky_relu_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "leaky_relu_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("negative_slope"), negative_slope);
  }
  Type::leaky_relu_out(output, self, negative_slope);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::leaky_relu(const Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("leaky_relu");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "leaky_relu", { self } );
    setattr(trace_info.n, jit::Symbol("negative_slope"), negative_slope);
  }
  auto output = Type::leaky_relu(self, negative_slope);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::leaky_relu_forward_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("leaky_relu_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("leaky_relu_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("leaky_relu_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "leaky_relu_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("negative_slope"), negative_slope);
  }
  baseType->leaky_relu_forward_out(output_, self_, negative_slope);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::leaky_relu_forward(const Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("leaky_relu_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LeakyReluBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<LeakyReluBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->negative_slope = negative_slope;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "leaky_relu_forward", { self } );
    setattr(trace_info.n, jit::Symbol("negative_slope"), negative_slope);
  }
  auto output = as_variable(baseType->leaky_relu_forward(self_, negative_slope));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("leaky_relu_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("leaky_relu_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("leaky_relu_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "leaky_relu_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("negative_slope"), negative_slope);
  }
  baseType->leaky_relu_backward_out(grad_input_, grad_output_, self_, negative_slope);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("leaky_relu_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<LeakyReluBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<LeakyReluBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->negative_slope = negative_slope;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "leaky_relu_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("negative_slope"), negative_slope);
  }
  auto grad_input = as_variable(baseType->leaky_relu_backward(grad_output_, self_, negative_slope));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::leaky_relu_(Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("leaky_relu_");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "leaky_relu", { self } );
    setattr(trace_info.n, jit::Symbol("negative_slope"), negative_slope);
  }
  Type::leaky_relu_(self, negative_slope);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::leaky_relu_forward_(Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("leaky_relu_forward_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LeakyReluBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<LeakyReluBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->negative_slope = negative_slope;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "leaky_relu_forward", { self } );
    setattr(trace_info.n, jit::Symbol("negative_slope"), negative_slope);
  }
  baseType->leaky_relu_forward_(self_, negative_slope);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  if (grad_fn) {
    grad_fn->output_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::log_sigmoid_out(Tensor & output, const Tensor & self) const {
  profiler::RecordFunction profiler("log_sigmoid_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "log_sigmoid_out", { output, self } );
  
  }
  Type::log_sigmoid_out(output, self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::log_sigmoid(const Tensor & self) const {
  profiler::RecordFunction profiler("log_sigmoid");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "log_sigmoid", { self } );
  
  }
  auto output = Type::log_sigmoid(self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &> VariableType::log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self) const {
  profiler::RecordFunction profiler("log_sigmoid_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& buffer_ = unpack(buffer, "buffer", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("log_sigmoid_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("log_sigmoid_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, buffer, self )) {
    trace_info = jit::tracer::preRecordTrace( "log_sigmoid_forward_out", { output, buffer, self } );
  
  }
  baseType->log_sigmoid_forward_out(output_, buffer_, self_);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, buffer} );
  }
  return std::forward_as_tuple(output, buffer);
}
std::tuple<Tensor,Tensor> VariableType::log_sigmoid_forward(const Tensor & self) const {
  profiler::RecordFunction profiler("log_sigmoid_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LogSigmoidBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<LogSigmoidBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  Tensor output;
  Tensor buffer;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "log_sigmoid_forward", { self } );
  
  }
  std::tie(output, buffer) = as_variable(baseType->log_sigmoid_forward(self_));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, buffer } );
  }
  if (grad_fn) {
    grad_fn->buffer_ = SavedVariable(buffer, true);
  }
  return std::make_tuple(std::move(output), std::move(buffer));
}
Tensor & VariableType::log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
  profiler::RecordFunction profiler("log_sigmoid_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& buffer_ = unpack(buffer, "buffer", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, buffer )) {
    throw_error_out_requires_grad("log_sigmoid_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("log_sigmoid_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, buffer )) {
    trace_info = jit::tracer::preRecordTrace( "log_sigmoid_backward_out", { grad_input, grad_output, self, buffer } );
  
  }
  baseType->log_sigmoid_backward_out(grad_input_, grad_output_, self_, buffer_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
  profiler::RecordFunction profiler("log_sigmoid_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& buffer_ = unpack(buffer, "buffer", 2);
  check_no_requires_grad(buffer, "buffer");
  std::shared_ptr<LogSigmoidBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<LogSigmoidBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->buffer_ = SavedVariable(buffer, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, buffer )) {
    trace_info = jit::tracer::preRecordTrace( "log_sigmoid_backward", { grad_output, self, buffer } );
  
  }
  auto grad_input = as_variable(baseType->log_sigmoid_backward(grad_output_, self_, buffer_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::log_softmax_out(Tensor & output, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("log_softmax_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "log_softmax_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  Type::log_softmax_out(output, self, dim);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::log_softmax(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("log_softmax");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "log_softmax", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto output = Type::log_softmax(self, dim);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::log_softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("log_softmax_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("log_softmax_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("log_softmax_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "log_softmax_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->log_softmax_forward_out(output_, self_, dim);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::log_softmax_forward(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("log_softmax_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LogSoftmaxBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<LogSoftmaxBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "log_softmax_forward", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto output = as_variable(baseType->log_softmax_forward(self_, dim));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  if (grad_fn) {
    grad_fn->output_ = SavedVariable(output, true);
  }
  return output;
}
Tensor & VariableType::log_softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
  profiler::RecordFunction profiler("log_softmax_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& output_ = unpack(output, "output", 4);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, output )) {
    throw_error_out_requires_grad("log_softmax_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("log_softmax_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, output )) {
    trace_info = jit::tracer::preRecordTrace( "log_softmax_backward_out", { grad_input, grad_output, self, output } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->log_softmax_backward_out(grad_input_, grad_output_, self_, dim, output_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::log_softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
  profiler::RecordFunction profiler("log_softmax_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& output_ = unpack(output, "output", 3);
  std::shared_ptr<LogSoftmaxBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<LogSoftmaxBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->dim = dim;
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, output )) {
    trace_info = jit::tracer::preRecordTrace( "log_softmax_backward", { grad_output, self, output } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto grad_input = as_variable(baseType->log_softmax_backward(grad_output_, self_, dim, output_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::prelu_out(Tensor & output, const Tensor & self, const Tensor & weight) const {
  profiler::RecordFunction profiler("prelu_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, weight )) {
    trace_info = jit::tracer::preRecordTrace( "prelu_out", { output, self, weight } );
  
  }
  Type::prelu_out(output, self, weight);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::prelu(const Tensor & self, const Tensor & weight) const {
  profiler::RecordFunction profiler("prelu");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight )) {
    trace_info = jit::tracer::preRecordTrace( "prelu", { self, weight } );
  
  }
  auto output = Type::prelu(self, weight);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::prelu_forward_out(Tensor & output, const Tensor & self, const Tensor & weight) const {
  profiler::RecordFunction profiler("prelu_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight )) {
    throw_error_out_requires_grad("prelu_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("prelu_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, weight )) {
    trace_info = jit::tracer::preRecordTrace( "prelu_forward_out", { output, self, weight } );
  
  }
  baseType->prelu_forward_out(output_, self_, weight_);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::prelu_forward(const Tensor & self, const Tensor & weight) const {
  profiler::RecordFunction profiler("prelu_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  std::shared_ptr<PreluBackward> grad_fn;
  if (compute_requires_grad( self, weight )) {
    grad_fn = std::make_shared<PreluBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight )) {
    trace_info = jit::tracer::preRecordTrace( "prelu_forward", { self, weight } );
  
  }
  auto output = as_variable(baseType->prelu_forward(self_, weight_));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &> VariableType::prelu_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight) const {
  profiler::RecordFunction profiler("prelu_backward_out");
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto& grad_output_ = unpack(grad_output, "grad_output", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    throw_error_out_requires_grad("prelu_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight )) {
    throw_error_out_requires_grad("prelu_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_weight, grad_output, self, weight )) {
    trace_info = jit::tracer::preRecordTrace( "prelu_backward_out", { grad_input, grad_weight, grad_output, self, weight } );
  
  }
  baseType->prelu_backward_out(grad_input_, grad_weight_, grad_output_, self_, weight_);
  increment_version(grad_input);
  increment_version(grad_weight);
  rebase_history({ grad_input, grad_weight }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input, grad_weight} );
  }
  return std::forward_as_tuple(grad_input, grad_weight);
}
std::tuple<Tensor,Tensor> VariableType::prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, std::array<bool,2> output_mask) const {
  profiler::RecordFunction profiler("prelu_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<PreluBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::make_shared<PreluBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  Tensor grad_input;
  Tensor grad_weight;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, weight )) {
    trace_info = jit::tracer::preRecordTrace( "prelu_backward", { grad_output, self, weight } );
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(grad_input, grad_weight) = as_variable(baseType->prelu_backward(grad_output_, self_, weight_, output_mask));
  set_history({ grad_input, grad_weight }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input, grad_weight } );
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight));
}
Tensor & VariableType::rrelu_with_noise_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("rrelu_with_noise_out");
  Type::rrelu_with_noise_out(output, self, noise, lower, upper, training, generator);
  return output;
}
Tensor VariableType::rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("rrelu_with_noise");
  auto output = Type::rrelu_with_noise(self, noise, lower, upper, training, generator);
  return output;
}
Tensor & VariableType::rrelu_with_noise_forward_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("rrelu_with_noise_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& noise_ = unpack(noise, "noise", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, noise )) {
    throw_error_out_requires_grad("rrelu_with_noise_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("rrelu_with_noise_forward");
  }
  baseType->rrelu_with_noise_forward_out(output_, self_, noise_, lower, upper, training, generator);
  increment_version(output);
  rebase_history(output, grad_fn);
  return output;
}
Tensor VariableType::rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("rrelu_with_noise_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& noise_ = unpack(noise, "noise", 1);
  check_no_requires_grad(noise, "noise");
  std::shared_ptr<RreluWithNoiseBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RreluWithNoiseBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->noise_ = SavedVariable(noise, false);
    grad_fn->lower = lower;
    grad_fn->upper = upper;
    grad_fn->training = training;
  }
  auto output = as_variable(baseType->rrelu_with_noise_forward(self_, noise_, lower, upper, training, generator));
  set_history(output, grad_fn);
  return output;
}
Tensor & VariableType::rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
  profiler::RecordFunction profiler("rrelu_with_noise_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& noise_ = unpack(noise, "noise", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, noise )) {
    throw_error_out_requires_grad("rrelu_with_noise_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("rrelu_with_noise_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, noise )) {
    trace_info = jit::tracer::preRecordTrace( "rrelu_with_noise_backward_out", { grad_input, grad_output, self, noise } );
    setattr(trace_info.n, jit::Symbol("lower"), lower);
    setattr(trace_info.n, jit::Symbol("upper"), upper);
    setattr(trace_info.n, jit::Symbol("training"), training);
  }
  baseType->rrelu_with_noise_backward_out(grad_input_, grad_output_, self_, noise_, lower, upper, training);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
  profiler::RecordFunction profiler("rrelu_with_noise_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& noise_ = unpack(noise, "noise", 2);
  check_no_requires_grad(noise, "noise");
  std::shared_ptr<RreluWithNoiseBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<RreluWithNoiseBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->noise_ = SavedVariable(noise, false);
    grad_fn->lower = lower;
    grad_fn->upper = upper;
    grad_fn->training = training;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, noise )) {
    trace_info = jit::tracer::preRecordTrace( "rrelu_with_noise_backward", { grad_output, self, noise } );
    setattr(trace_info.n, jit::Symbol("lower"), lower);
    setattr(trace_info.n, jit::Symbol("upper"), upper);
    setattr(trace_info.n, jit::Symbol("training"), training);
  }
  auto grad_input = as_variable(baseType->rrelu_with_noise_backward(grad_output_, self_, noise_, lower, upper, training));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("rrelu_with_noise_");
  Type::rrelu_with_noise_(self, noise, lower, upper, training, generator);
  return self;
}
Tensor & VariableType::rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("rrelu_with_noise_forward_");
  auto& self_ = unpack(self, "self", 0);
  auto& noise_ = unpack(noise, "noise", 1);
  check_inplace(self);
  check_no_requires_grad(noise, "noise");
  std::shared_ptr<RreluWithNoiseBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RreluWithNoiseBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->noise_ = SavedVariable(noise, false);
    grad_fn->lower = lower;
    grad_fn->upper = upper;
    grad_fn->training = training;
  }
  baseType->rrelu_with_noise_forward_(self_, noise_, lower, upper, training, generator);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (grad_fn) {
    grad_fn->output_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::softmax_out(Tensor & output, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("softmax_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "softmax_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  Type::softmax_out(output, self, dim);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::softmax(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("softmax");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "softmax", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto output = Type::softmax(self, dim);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("softmax_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("softmax_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("softmax_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "softmax_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->softmax_forward_out(output_, self_, dim);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::softmax_forward(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("softmax_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SoftmaxBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SoftmaxBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "softmax_forward", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto output = as_variable(baseType->softmax_forward(self_, dim));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  if (grad_fn) {
    grad_fn->output_ = SavedVariable(output, true);
  }
  return output;
}
Tensor & VariableType::softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
  profiler::RecordFunction profiler("softmax_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& output_ = unpack(output, "output", 4);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, output )) {
    throw_error_out_requires_grad("softmax_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("softmax_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, output )) {
    trace_info = jit::tracer::preRecordTrace( "softmax_backward_out", { grad_input, grad_output, self, output } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->softmax_backward_out(grad_input_, grad_output_, self_, dim, output_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
  profiler::RecordFunction profiler("softmax_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& output_ = unpack(output, "output", 3);
  std::shared_ptr<SoftmaxBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<SoftmaxBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, output )) {
    trace_info = jit::tracer::preRecordTrace( "softmax_backward", { grad_output, self, output } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto grad_input = as_variable(baseType->softmax_backward(grad_output_, self_, dim, output_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::softplus_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
  profiler::RecordFunction profiler("softplus_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "softplus_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
  }
  Type::softplus_out(output, self, beta, threshold);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::softplus(const Tensor & self, Scalar beta, Scalar threshold) const {
  profiler::RecordFunction profiler("softplus");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "softplus", { self } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
  }
  auto output = Type::softplus(self, beta, threshold);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::softplus_forward_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
  profiler::RecordFunction profiler("softplus_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("softplus_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("softplus_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "softplus_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
  }
  baseType->softplus_forward_out(output_, self_, beta, threshold);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::softplus_forward(const Tensor & self, Scalar beta, Scalar threshold) const {
  profiler::RecordFunction profiler("softplus_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SoftplusBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SoftplusBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->beta = beta;
    grad_fn->threshold = threshold;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "softplus_forward", { self } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
  }
  auto output = as_variable(baseType->softplus_forward(self_, beta, threshold));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  if (grad_fn) {
    grad_fn->output_ = SavedVariable(output, true);
  }
  return output;
}
Tensor & VariableType::softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
  profiler::RecordFunction profiler("softplus_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& output_ = unpack(output, "output", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, output )) {
    throw_error_out_requires_grad("softplus_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("softplus_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, output )) {
    trace_info = jit::tracer::preRecordTrace( "softplus_backward_out", { grad_input, grad_output, self, output } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
  }
  baseType->softplus_backward_out(grad_input_, grad_output_, self_, beta, threshold, output_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
  profiler::RecordFunction profiler("softplus_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& output_ = unpack(output, "output", 4);
  std::shared_ptr<SoftplusBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<SoftplusBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->beta = beta;
    grad_fn->threshold = threshold;
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, output )) {
    trace_info = jit::tracer::preRecordTrace( "softplus_backward", { grad_output, self, output } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
  }
  auto grad_input = as_variable(baseType->softplus_backward(grad_output_, self_, beta, threshold, output_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::softshrink_out(Tensor & output, const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("softshrink_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "softshrink_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("lambd"), lambd);
  }
  Type::softshrink_out(output, self, lambd);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::softshrink(const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("softshrink");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "softshrink", { self } );
    setattr(trace_info.n, jit::Symbol("lambd"), lambd);
  }
  auto output = Type::softshrink(self, lambd);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::softshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("softshrink_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("softshrink_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("softshrink_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "softshrink_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("lambd"), lambd);
  }
  baseType->softshrink_forward_out(output_, self_, lambd);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::softshrink_forward(const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("softshrink_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SoftshrinkBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SoftshrinkBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->lambd = lambd;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "softshrink_forward", { self } );
    setattr(trace_info.n, jit::Symbol("lambd"), lambd);
  }
  auto output = as_variable(baseType->softshrink_forward(self_, lambd));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("softshrink_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("softshrink_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("softshrink_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "softshrink_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("lambd"), lambd);
  }
  baseType->softshrink_backward_out(grad_input_, grad_output_, self_, lambd);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("softshrink_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<SoftshrinkBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<SoftshrinkBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->lambd = lambd;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "softshrink_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("lambd"), lambd);
  }
  auto grad_input = as_variable(baseType->softshrink_backward(grad_output_, self_, lambd));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::threshold_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) const {
  profiler::RecordFunction profiler("threshold_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "threshold_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  Type::threshold_out(output, self, threshold, value);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::threshold(const Tensor & self, Scalar threshold, Scalar value) const {
  profiler::RecordFunction profiler("threshold");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "threshold", { self } );
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  auto output = Type::threshold(self, threshold, value);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::threshold_forward_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) const {
  profiler::RecordFunction profiler("threshold_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("threshold_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("threshold_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "threshold_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  baseType->threshold_forward_out(output_, self_, threshold, value);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::threshold_forward(const Tensor & self, Scalar threshold, Scalar value) const {
  profiler::RecordFunction profiler("threshold_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ThresholdBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ThresholdBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->threshold = threshold;
    grad_fn->value = value;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "threshold_forward", { self } );
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  auto output = as_variable(baseType->threshold_forward(self_, threshold, value));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::threshold_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
  profiler::RecordFunction profiler("threshold_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("threshold_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("threshold_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "threshold_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  baseType->threshold_backward_out(grad_input_, grad_output_, self_, threshold, value);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
  profiler::RecordFunction profiler("threshold_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ThresholdBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<ThresholdBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->threshold = threshold;
    grad_fn->value = value;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "threshold_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  auto grad_input = as_variable(baseType->threshold_backward(grad_output_, self_, threshold, value));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::threshold_(Tensor & self, Scalar threshold, Scalar value) const {
  profiler::RecordFunction profiler("threshold_");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "threshold", { self } );
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  Type::threshold_(self, threshold, value);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::threshold_forward_(Tensor & self, Scalar threshold, Scalar value) const {
  profiler::RecordFunction profiler("threshold_forward_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ThresholdBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ThresholdBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->threshold = threshold;
    grad_fn->value = value;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "threshold_forward", { self } );
    setattr(trace_info.n, jit::Symbol("threshold"), threshold);
    setattr(trace_info.n, jit::Symbol("value"), value);
  }
  baseType->threshold_forward_(self_, threshold, value);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  if (grad_fn) {
    grad_fn->output_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::adaptive_avg_pool2d_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_avg_pool2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_avg_pool2d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  Type::adaptive_avg_pool2d_out(output, self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::adaptive_avg_pool2d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_avg_pool2d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_avg_pool2d", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto output = Type::adaptive_avg_pool2d(self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::adaptive_avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_avg_pool2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("adaptive_avg_pool2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("adaptive_avg_pool2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_avg_pool2d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  baseType->adaptive_avg_pool2d_forward_out(output_, self_, output_size);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::adaptive_avg_pool2d_forward(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_avg_pool2d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AdaptiveAvgPool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AdaptiveAvgPool2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_avg_pool2d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto output = as_variable(baseType->adaptive_avg_pool2d_forward(self_, output_size));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::adaptive_avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
  profiler::RecordFunction profiler("adaptive_avg_pool2d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("adaptive_avg_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("adaptive_avg_pool2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_avg_pool2d_backward_out", { grad_input, grad_output, self } );
  
  }
  baseType->adaptive_avg_pool2d_backward_out(grad_input_, grad_output_, self_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) const {
  profiler::RecordFunction profiler("adaptive_avg_pool2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<AdaptiveAvgPool2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<AdaptiveAvgPool2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_avg_pool2d_backward", { grad_output, self } );
  
  }
  auto grad_input = as_variable(baseType->adaptive_avg_pool2d_backward(grad_output_, self_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::adaptive_avg_pool3d_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_avg_pool3d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_avg_pool3d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  Type::adaptive_avg_pool3d_out(output, self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::adaptive_avg_pool3d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_avg_pool3d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_avg_pool3d", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto output = Type::adaptive_avg_pool3d(self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::adaptive_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_avg_pool3d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("adaptive_avg_pool3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("adaptive_avg_pool3d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_avg_pool3d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  baseType->adaptive_avg_pool3d_forward_out(output_, self_, output_size);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::adaptive_avg_pool3d_forward(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_avg_pool3d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AdaptiveAvgPool3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AdaptiveAvgPool3DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_avg_pool3d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto output = as_variable(baseType->adaptive_avg_pool3d_forward(self_, output_size));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
  profiler::RecordFunction profiler("adaptive_avg_pool3d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("adaptive_avg_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("adaptive_avg_pool3d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_avg_pool3d_backward_out", { grad_input, grad_output, self } );
  
  }
  baseType->adaptive_avg_pool3d_backward_out(grad_input_, grad_output_, self_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) const {
  profiler::RecordFunction profiler("adaptive_avg_pool3d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<AdaptiveAvgPool3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<AdaptiveAvgPool3DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_avg_pool3d_backward", { grad_output, self } );
  
  }
  auto grad_input = as_variable(baseType->adaptive_avg_pool3d_backward(grad_output_, self_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::adaptive_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_max_pool2d_out", { output, indices, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  Type::adaptive_max_pool2d_out(output, indices, self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, indices} );
  }
  return std::forward_as_tuple(output, indices);
}
std::tuple<Tensor,Tensor> VariableType::adaptive_max_pool2d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool2d");
  Tensor output;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_max_pool2d", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  std::tie(output, indices) = Type::adaptive_max_pool2d(self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, indices } );
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
std::tuple<Tensor &,Tensor &> VariableType::adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("adaptive_max_pool2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("adaptive_max_pool2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_max_pool2d_forward_out", { output, indices, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  baseType->adaptive_max_pool2d_forward_out(output_, indices_, self_, output_size);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, indices} );
  }
  return std::forward_as_tuple(output, indices);
}
std::tuple<Tensor,Tensor> VariableType::adaptive_max_pool2d_forward(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool2d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AdaptiveMaxPool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AdaptiveMaxPool2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  Tensor output;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_max_pool2d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  std::tie(output, indices) = as_variable(baseType->adaptive_max_pool2d_forward(self_, output_size));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, indices } );
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
Tensor & VariableType::adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
  profiler::RecordFunction profiler("adaptive_max_pool2d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("adaptive_max_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("adaptive_max_pool2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_max_pool2d_backward_out", { grad_input, grad_output, self, indices } );
  
  }
  baseType->adaptive_max_pool2d_backward_out(grad_input_, grad_output_, self_, indices_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
  profiler::RecordFunction profiler("adaptive_max_pool2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<AdaptiveMaxPool2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<AdaptiveMaxPool2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_max_pool2d_backward", { grad_output, self, indices } );
  
  }
  auto grad_input = as_variable(baseType->adaptive_max_pool2d_backward(grad_output_, self_, indices_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::adaptive_max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool3d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_max_pool3d_out", { output, indices, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  Type::adaptive_max_pool3d_out(output, indices, self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, indices} );
  }
  return std::forward_as_tuple(output, indices);
}
std::tuple<Tensor,Tensor> VariableType::adaptive_max_pool3d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool3d");
  Tensor output;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_max_pool3d", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  std::tie(output, indices) = Type::adaptive_max_pool3d(self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, indices } );
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
std::tuple<Tensor &,Tensor &> VariableType::adaptive_max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool3d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("adaptive_max_pool3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("adaptive_max_pool3d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_max_pool3d_forward_out", { output, indices, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  baseType->adaptive_max_pool3d_forward_out(output_, indices_, self_, output_size);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, indices} );
  }
  return std::forward_as_tuple(output, indices);
}
std::tuple<Tensor,Tensor> VariableType::adaptive_max_pool3d_forward(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool3d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AdaptiveMaxPool3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AdaptiveMaxPool3DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  Tensor output;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_max_pool3d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  std::tie(output, indices) = as_variable(baseType->adaptive_max_pool3d_forward(self_, output_size));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, indices } );
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
Tensor & VariableType::adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
  profiler::RecordFunction profiler("adaptive_max_pool3d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("adaptive_max_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("adaptive_max_pool3d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_max_pool3d_backward_out", { grad_input, grad_output, self, indices } );
  
  }
  baseType->adaptive_max_pool3d_backward_out(grad_input_, grad_output_, self_, indices_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
  profiler::RecordFunction profiler("adaptive_max_pool3d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<AdaptiveMaxPool3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<AdaptiveMaxPool3DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_max_pool3d_backward", { grad_output, self, indices } );
  
  }
  auto grad_input = as_variable(baseType->adaptive_max_pool3d_backward(grad_output_, self_, indices_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::avg_pool2d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "avg_pool2d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
    setattr(trace_info.n, jit::Symbol("count_include_pad"), count_include_pad);
  }
  Type::avg_pool2d_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::avg_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool2d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "avg_pool2d", { self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
    setattr(trace_info.n, jit::Symbol("count_include_pad"), count_include_pad);
  }
  auto output = Type::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("avg_pool2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("avg_pool2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "avg_pool2d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
    setattr(trace_info.n, jit::Symbol("count_include_pad"), count_include_pad);
  }
  baseType->avg_pool2d_forward_out(output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::avg_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool2d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AvgPool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AvgPool2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->ceil_mode = ceil_mode;
    grad_fn->count_include_pad = count_include_pad;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "avg_pool2d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
    setattr(trace_info.n, jit::Symbol("count_include_pad"), count_include_pad);
  }
  auto output = as_variable(baseType->avg_pool2d_forward(self_, kernel_size, stride, padding, ceil_mode, count_include_pad));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool2d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("avg_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("avg_pool2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "avg_pool2d_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
    setattr(trace_info.n, jit::Symbol("count_include_pad"), count_include_pad);
  }
  baseType->avg_pool2d_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<AvgPool2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<AvgPool2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->kernel_size = kernel_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->ceil_mode = ceil_mode;
    grad_fn->count_include_pad = count_include_pad;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "avg_pool2d_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
    setattr(trace_info.n, jit::Symbol("count_include_pad"), count_include_pad);
  }
  auto grad_input = as_variable(baseType->avg_pool2d_backward(grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::avg_pool3d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool3d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "avg_pool3d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
    setattr(trace_info.n, jit::Symbol("count_include_pad"), count_include_pad);
  }
  Type::avg_pool3d_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::avg_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool3d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "avg_pool3d", { self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
    setattr(trace_info.n, jit::Symbol("count_include_pad"), count_include_pad);
  }
  auto output = Type::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool3d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("avg_pool3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("avg_pool3d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "avg_pool3d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
    setattr(trace_info.n, jit::Symbol("count_include_pad"), count_include_pad);
  }
  baseType->avg_pool3d_forward_out(output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::avg_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool3d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AvgPool3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<AvgPool3DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->ceil_mode = ceil_mode;
    grad_fn->count_include_pad = count_include_pad;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "avg_pool3d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
    setattr(trace_info.n, jit::Symbol("count_include_pad"), count_include_pad);
  }
  auto output = as_variable(baseType->avg_pool3d_forward(self_, kernel_size, stride, padding, ceil_mode, count_include_pad));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool3d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("avg_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("avg_pool3d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "avg_pool3d_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
    setattr(trace_info.n, jit::Symbol("count_include_pad"), count_include_pad);
  }
  baseType->avg_pool3d_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool3d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<AvgPool3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<AvgPool3DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->kernel_size = kernel_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->ceil_mode = ceil_mode;
    grad_fn->count_include_pad = count_include_pad;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "avg_pool3d_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
    setattr(trace_info.n, jit::Symbol("count_include_pad"), count_include_pad);
  }
  auto grad_input = as_variable(baseType->avg_pool3d_backward(grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::fractional_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
  profiler::RecordFunction profiler("fractional_max_pool2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, indices, self, random_samples )) {
    trace_info = jit::tracer::preRecordTrace( "fractional_max_pool2d_out", { output, indices, self, random_samples } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  Type::fractional_max_pool2d_out(output, indices, self, kernel_size, output_size, random_samples);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, indices} );
  }
  return std::forward_as_tuple(output, indices);
}
std::tuple<Tensor,Tensor> VariableType::fractional_max_pool2d(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
  profiler::RecordFunction profiler("fractional_max_pool2d");
  Tensor output;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, random_samples )) {
    trace_info = jit::tracer::preRecordTrace( "fractional_max_pool2d", { self, random_samples } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  std::tie(output, indices) = Type::fractional_max_pool2d(self, kernel_size, output_size, random_samples);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, indices } );
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
std::tuple<Tensor &,Tensor &> VariableType::fractional_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
  profiler::RecordFunction profiler("fractional_max_pool2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& random_samples_ = unpack(random_samples, "random_samples", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, random_samples )) {
    throw_error_out_requires_grad("fractional_max_pool2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("fractional_max_pool2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, indices, self, random_samples )) {
    trace_info = jit::tracer::preRecordTrace( "fractional_max_pool2d_forward_out", { output, indices, self, random_samples } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  baseType->fractional_max_pool2d_forward_out(output_, indices_, self_, kernel_size, output_size, random_samples_);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, indices} );
  }
  return std::forward_as_tuple(output, indices);
}
std::tuple<Tensor,Tensor> VariableType::fractional_max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
  profiler::RecordFunction profiler("fractional_max_pool2d_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& random_samples_ = unpack(random_samples, "random_samples", 3);
  check_no_requires_grad(random_samples, "random_samples");
  std::shared_ptr<FractionalMaxPool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<FractionalMaxPool2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size;
    grad_fn->output_size = output_size;
  }
  Tensor output;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, random_samples )) {
    trace_info = jit::tracer::preRecordTrace( "fractional_max_pool2d_forward", { self, random_samples } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  std::tie(output, indices) = as_variable(baseType->fractional_max_pool2d_forward(self_, kernel_size, output_size, random_samples_));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, indices } );
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
Tensor & VariableType::fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
  profiler::RecordFunction profiler("fractional_max_pool2d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("fractional_max_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("fractional_max_pool2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "fractional_max_pool2d_backward_out", { grad_input, grad_output, self, indices } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  baseType->fractional_max_pool2d_backward_out(grad_input_, grad_output_, self_, kernel_size, output_size, indices_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
  profiler::RecordFunction profiler("fractional_max_pool2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 4);
  std::shared_ptr<FractionalMaxPool2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<FractionalMaxPool2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "fractional_max_pool2d_backward", { grad_output, self, indices } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto grad_input = as_variable(baseType->fractional_max_pool2d_backward(grad_output_, self_, kernel_size, output_size, indices_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "max_pool2d_out", { output, indices, self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
  }
  Type::max_pool2d_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, indices} );
  }
  return std::forward_as_tuple(output, indices);
}
std::tuple<Tensor,Tensor> VariableType::max_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool2d");
  Tensor output;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "max_pool2d", { self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
  }
  std::tie(output, indices) = Type::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, indices } );
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
std::tuple<Tensor &,Tensor &> VariableType::max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max_pool2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("max_pool2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "max_pool2d_forward_out", { output, indices, self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
  }
  baseType->max_pool2d_forward_out(output_, indices_, self_, kernel_size, stride, padding, dilation, ceil_mode);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, indices} );
  }
  return std::forward_as_tuple(output, indices);
}
std::tuple<Tensor,Tensor> VariableType::max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool2d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MaxPool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MaxPool2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->dilation = dilation;
    grad_fn->ceil_mode = ceil_mode;
  }
  Tensor output;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "max_pool2d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
  }
  std::tie(output, indices) = as_variable(baseType->max_pool2d_forward(self_, kernel_size, stride, padding, dilation, ceil_mode));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, indices } );
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
Tensor & VariableType::max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
  profiler::RecordFunction profiler("max_pool2d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 8);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("max_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("max_pool2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_pool2d_backward_out", { grad_input, grad_output, self, indices } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
  }
  baseType->max_pool2d_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
  profiler::RecordFunction profiler("max_pool2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 7);
  std::shared_ptr<MaxPool2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<MaxPool2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_pool2d_backward", { grad_output, self, indices } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
  }
  auto grad_input = as_variable(baseType->max_pool2d_backward(grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool3d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "max_pool3d_out", { output, indices, self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
  }
  Type::max_pool3d_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, indices} );
  }
  return std::forward_as_tuple(output, indices);
}
std::tuple<Tensor,Tensor> VariableType::max_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool3d");
  Tensor output;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "max_pool3d", { self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
  }
  std::tie(output, indices) = Type::max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, indices } );
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
std::tuple<Tensor &,Tensor &> VariableType::max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool3d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max_pool3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("max_pool3d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, indices, self )) {
    trace_info = jit::tracer::preRecordTrace( "max_pool3d_forward_out", { output, indices, self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
  }
  baseType->max_pool3d_forward_out(output_, indices_, self_, kernel_size, stride, padding, dilation, ceil_mode);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, indices} );
  }
  return std::forward_as_tuple(output, indices);
}
std::tuple<Tensor,Tensor> VariableType::max_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool3d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MaxPool3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MaxPool3DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->dilation = dilation;
    grad_fn->ceil_mode = ceil_mode;
  }
  Tensor output;
  Tensor indices;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "max_pool3d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
  }
  std::tie(output, indices) = as_variable(baseType->max_pool3d_forward(self_, kernel_size, stride, padding, dilation, ceil_mode));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, indices } );
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
Tensor & VariableType::max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
  profiler::RecordFunction profiler("max_pool3d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 8);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("max_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("max_pool3d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_pool3d_backward_out", { grad_input, grad_output, self, indices } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
  }
  baseType->max_pool3d_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
  profiler::RecordFunction profiler("max_pool3d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 7);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<Error>("the derivative for max_pool3d_backward is not implemented");
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_pool3d_backward", { grad_output, self, indices } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
  }
  auto grad_input = as_variable(baseType->max_pool3d_backward(grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::max_unpool2d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) const {
  profiler::RecordFunction profiler("max_unpool2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_unpool2d_out", { output, self, indices } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  Type::max_unpool2d_out(output, self, indices, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::max_unpool2d(const Tensor & self, const Tensor & indices, IntList output_size) const {
  profiler::RecordFunction profiler("max_unpool2d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_unpool2d", { self, indices } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto output = Type::max_unpool2d(self, indices, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::max_unpool2d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) const {
  profiler::RecordFunction profiler("max_unpool2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max_unpool2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("max_unpool2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_unpool2d_forward_out", { output, self, indices } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  baseType->max_unpool2d_forward_out(output_, self_, indices_, output_size);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::max_unpool2d_forward(const Tensor & self, const Tensor & indices, IntList output_size) const {
  profiler::RecordFunction profiler("max_unpool2d_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<MaxUnpool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MaxUnpool2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->output_size = output_size;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_unpool2d_forward", { self, indices } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto output = as_variable(baseType->max_unpool2d_forward(self_, indices_, output_size));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
  profiler::RecordFunction profiler("max_unpool2d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("max_unpool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("max_unpool2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_unpool2d_backward_out", { grad_input, grad_output, self, indices } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  baseType->max_unpool2d_backward_out(grad_input_, grad_output_, self_, indices_, output_size);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
  profiler::RecordFunction profiler("max_unpool2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<MaxUnpool2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<MaxUnpool2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->output_size = output_size;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_unpool2d_backward", { grad_output, self, indices } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto grad_input = as_variable(baseType->max_unpool2d_backward(grad_output_, self_, indices_, output_size));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::max_unpool3d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("max_unpool3d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_unpool3d_out", { output, self, indices } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  Type::max_unpool3d_out(output, self, indices, output_size, stride, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::max_unpool3d(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("max_unpool3d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_unpool3d", { self, indices } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = Type::max_unpool3d(self, indices, output_size, stride, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::max_unpool3d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("max_unpool3d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max_unpool3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("max_unpool3d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_unpool3d_forward_out", { output, self, indices } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->max_unpool3d_forward_out(output_, self_, indices_, output_size, stride, padding);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::max_unpool3d_forward(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("max_unpool3d_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<MaxUnpool3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<MaxUnpool3DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->output_size = output_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_unpool3d_forward", { self, indices } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = as_variable(baseType->max_unpool3d_forward(self_, indices_, output_size, stride, padding));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("max_unpool3d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("max_unpool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("max_unpool3d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_unpool3d_backward_out", { grad_input, grad_output, self, indices } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->max_unpool3d_backward_out(grad_input_, grad_output_, self_, indices_, output_size, stride, padding);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("max_unpool3d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<Error>("the derivative for max_unpool3d_backward is not implemented");
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "max_unpool3d_backward", { grad_output, self, indices } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto grad_input = as_variable(baseType->max_unpool3d_backward(grad_output_, self_, indices_, output_size, stride, padding));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::reflection_pad1d_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad1d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "reflection_pad1d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  Type::reflection_pad1d_out(output, self, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::reflection_pad1d(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad1d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "reflection_pad1d", { self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = Type::reflection_pad1d(self, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::reflection_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad1d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("reflection_pad1d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("reflection_pad1d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "reflection_pad1d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->reflection_pad1d_forward_out(output_, self_, padding);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::reflection_pad1d_forward(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad1d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReflectionPad1DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ReflectionPad1DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "reflection_pad1d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = as_variable(baseType->reflection_pad1d_forward(self_, padding));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad1d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("reflection_pad1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("reflection_pad1d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "reflection_pad1d_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->reflection_pad1d_backward_out(grad_input_, grad_output_, self_, padding);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad1d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ReflectionPad1DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<ReflectionPad1DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "reflection_pad1d_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto grad_input = as_variable(baseType->reflection_pad1d_backward(grad_output_, self_, padding));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::reflection_pad2d_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "reflection_pad2d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  Type::reflection_pad2d_out(output, self, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::reflection_pad2d(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad2d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "reflection_pad2d", { self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = Type::reflection_pad2d(self, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::reflection_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("reflection_pad2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("reflection_pad2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "reflection_pad2d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->reflection_pad2d_forward_out(output_, self_, padding);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::reflection_pad2d_forward(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad2d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReflectionPad2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ReflectionPad2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "reflection_pad2d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = as_variable(baseType->reflection_pad2d_forward(self_, padding));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::reflection_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad2d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("reflection_pad2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("reflection_pad2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "reflection_pad2d_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->reflection_pad2d_backward_out(grad_input_, grad_output_, self_, padding);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ReflectionPad2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<ReflectionPad2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "reflection_pad2d_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto grad_input = as_variable(baseType->reflection_pad2d_backward(grad_output_, self_, padding));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::replication_pad1d_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad1d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad1d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  Type::replication_pad1d_out(output, self, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::replication_pad1d(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad1d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad1d", { self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = Type::replication_pad1d(self, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::replication_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad1d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("replication_pad1d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("replication_pad1d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad1d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->replication_pad1d_forward_out(output_, self_, padding);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::replication_pad1d_forward(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad1d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReplicationPad1DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ReplicationPad1DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad1d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = as_variable(baseType->replication_pad1d_forward(self_, padding));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::replication_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad1d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("replication_pad1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("replication_pad1d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad1d_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->replication_pad1d_backward_out(grad_input_, grad_output_, self_, padding);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad1d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ReplicationPad1DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<ReplicationPad1DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad1d_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto grad_input = as_variable(baseType->replication_pad1d_backward(grad_output_, self_, padding));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::replication_pad2d_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad2d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  Type::replication_pad2d_out(output, self, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::replication_pad2d(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad2d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad2d", { self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = Type::replication_pad2d(self, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::replication_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("replication_pad2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("replication_pad2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad2d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->replication_pad2d_forward_out(output_, self_, padding);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::replication_pad2d_forward(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad2d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReplicationPad2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ReplicationPad2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad2d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = as_variable(baseType->replication_pad2d_forward(self_, padding));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad2d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("replication_pad2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("replication_pad2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad2d_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->replication_pad2d_backward_out(grad_input_, grad_output_, self_, padding);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ReplicationPad2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<ReplicationPad2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad2d_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto grad_input = as_variable(baseType->replication_pad2d_backward(grad_output_, self_, padding));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::replication_pad3d_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad3d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad3d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  Type::replication_pad3d_out(output, self, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::replication_pad3d(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad3d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad3d", { self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = Type::replication_pad3d(self, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::replication_pad3d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad3d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("replication_pad3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("replication_pad3d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad3d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->replication_pad3d_forward_out(output_, self_, padding);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::replication_pad3d_forward(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad3d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReplicationPad3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ReplicationPad3DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad3d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = as_variable(baseType->replication_pad3d_forward(self_, padding));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad3d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("replication_pad3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("replication_pad3d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad3d_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->replication_pad3d_backward_out(grad_input_, grad_output_, self_, padding);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad3d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ReplicationPad3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<ReplicationPad3DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding;
    grad_fn->self_info = self;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "replication_pad3d_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto grad_input = as_variable(baseType->replication_pad3d_backward(grad_output_, self_, padding));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::upsample_linear1d_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_linear1d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_linear1d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  Type::upsample_linear1d_out(output, self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::upsample_linear1d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_linear1d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_linear1d", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto output = Type::upsample_linear1d(self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::upsample_linear1d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_linear1d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_linear1d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("upsample_linear1d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_linear1d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  baseType->upsample_linear1d_forward_out(output_, self_, output_size);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::upsample_linear1d_forward(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_linear1d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleLinear1DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<UpsampleLinear1DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->output_size = output_size;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_linear1d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto output = as_variable(baseType->upsample_linear1d_forward(self_, output_size));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size) const {
  profiler::RecordFunction profiler("upsample_linear1d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_linear1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_linear1d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_linear1d_backward_out", { grad_input, grad_output } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
    setattr(trace_info.n, jit::Symbol("input_size"), input_size);
  }
  baseType->upsample_linear1d_backward_out(grad_input_, grad_output_, output_size, input_size);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::upsample_linear1d_backward(const Tensor & grad_output, IntList output_size, IntList input_size) const {
  profiler::RecordFunction profiler("upsample_linear1d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleLinear1DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::make_shared<UpsampleLinear1DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_linear1d_backward", { grad_output } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
    setattr(trace_info.n, jit::Symbol("input_size"), input_size);
  }
  auto grad_input = as_variable(baseType->upsample_linear1d_backward(grad_output_, output_size, input_size));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::upsample_bilinear2d_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_bilinear2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_bilinear2d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  Type::upsample_bilinear2d_out(output, self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::upsample_bilinear2d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_bilinear2d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_bilinear2d", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto output = Type::upsample_bilinear2d(self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::upsample_bilinear2d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_bilinear2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_bilinear2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("upsample_bilinear2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_bilinear2d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  baseType->upsample_bilinear2d_forward_out(output_, self_, output_size);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::upsample_bilinear2d_forward(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_bilinear2d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleBilinear2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<UpsampleBilinear2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->output_size = output_size;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_bilinear2d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto output = as_variable(baseType->upsample_bilinear2d_forward(self_, output_size));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size) const {
  profiler::RecordFunction profiler("upsample_bilinear2d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_bilinear2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_bilinear2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_bilinear2d_backward_out", { grad_input, grad_output } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
    setattr(trace_info.n, jit::Symbol("input_size"), input_size);
  }
  baseType->upsample_bilinear2d_backward_out(grad_input_, grad_output_, output_size, input_size);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::upsample_bilinear2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size) const {
  profiler::RecordFunction profiler("upsample_bilinear2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleBilinear2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::make_shared<UpsampleBilinear2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_bilinear2d_backward", { grad_output } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
    setattr(trace_info.n, jit::Symbol("input_size"), input_size);
  }
  auto grad_input = as_variable(baseType->upsample_bilinear2d_backward(grad_output_, output_size, input_size));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::upsample_trilinear3d_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_trilinear3d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_trilinear3d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  Type::upsample_trilinear3d_out(output, self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::upsample_trilinear3d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_trilinear3d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_trilinear3d", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto output = Type::upsample_trilinear3d(self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::upsample_trilinear3d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_trilinear3d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_trilinear3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("upsample_trilinear3d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_trilinear3d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  baseType->upsample_trilinear3d_forward_out(output_, self_, output_size);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::upsample_trilinear3d_forward(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_trilinear3d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleTrilinear3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<UpsampleTrilinear3DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->output_size = output_size;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_trilinear3d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto output = as_variable(baseType->upsample_trilinear3d_forward(self_, output_size));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size) const {
  profiler::RecordFunction profiler("upsample_trilinear3d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_trilinear3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_trilinear3d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_trilinear3d_backward_out", { grad_input, grad_output } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
    setattr(trace_info.n, jit::Symbol("input_size"), input_size);
  }
  baseType->upsample_trilinear3d_backward_out(grad_input_, grad_output_, output_size, input_size);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::upsample_trilinear3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size) const {
  profiler::RecordFunction profiler("upsample_trilinear3d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleTrilinear3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::make_shared<UpsampleTrilinear3DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_trilinear3d_backward", { grad_output } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
    setattr(trace_info.n, jit::Symbol("input_size"), input_size);
  }
  auto grad_input = as_variable(baseType->upsample_trilinear3d_backward(grad_output_, output_size, input_size));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::upsample_nearest1d_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest1d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest1d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  Type::upsample_nearest1d_out(output, self, scale_factor);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::upsample_nearest1d(const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest1d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest1d", { self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  auto output = Type::upsample_nearest1d(self, scale_factor);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::upsample_nearest1d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest1d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_nearest1d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("upsample_nearest1d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest1d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  baseType->upsample_nearest1d_forward_out(output_, self_, scale_factor);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::upsample_nearest1d_forward(const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest1d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleNearest1DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<UpsampleNearest1DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->scale_factor = scale_factor;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest1d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  auto output = as_variable(baseType->upsample_nearest1d_forward(self_, scale_factor));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest1d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("upsample_nearest1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_nearest1d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest1d_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  baseType->upsample_nearest1d_backward_out(grad_input_, grad_output_, self_, scale_factor);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::upsample_nearest1d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest1d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<UpsampleNearest1DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<UpsampleNearest1DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->scale_factor = scale_factor;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest1d_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  auto grad_input = as_variable(baseType->upsample_nearest1d_backward(grad_output_, self_, scale_factor));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::upsample_nearest2d_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest2d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  Type::upsample_nearest2d_out(output, self, scale_factor);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::upsample_nearest2d(const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest2d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest2d", { self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  auto output = Type::upsample_nearest2d(self, scale_factor);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::upsample_nearest2d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_nearest2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("upsample_nearest2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest2d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  baseType->upsample_nearest2d_forward_out(output_, self_, scale_factor);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::upsample_nearest2d_forward(const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest2d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleNearest2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<UpsampleNearest2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->scale_factor = scale_factor;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest2d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  auto output = as_variable(baseType->upsample_nearest2d_forward(self_, scale_factor));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest2d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("upsample_nearest2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_nearest2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest2d_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  baseType->upsample_nearest2d_backward_out(grad_input_, grad_output_, self_, scale_factor);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::upsample_nearest2d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<UpsampleNearest2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<UpsampleNearest2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->scale_factor = scale_factor;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest2d_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  auto grad_input = as_variable(baseType->upsample_nearest2d_backward(grad_output_, self_, scale_factor));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::upsample_nearest3d_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest3d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest3d_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  Type::upsample_nearest3d_out(output, self, scale_factor);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::upsample_nearest3d(const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest3d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest3d", { self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  auto output = Type::upsample_nearest3d(self, scale_factor);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::upsample_nearest3d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest3d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_nearest3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("upsample_nearest3d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest3d_forward_out", { output, self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  baseType->upsample_nearest3d_forward_out(output_, self_, scale_factor);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::upsample_nearest3d_forward(const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest3d_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleNearest3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<UpsampleNearest3DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->scale_factor = scale_factor;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest3d_forward", { self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  auto output = as_variable(baseType->upsample_nearest3d_forward(self_, scale_factor));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest3d_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("upsample_nearest3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_nearest3d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest3d_backward_out", { grad_input, grad_output, self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  baseType->upsample_nearest3d_backward_out(grad_input_, grad_output_, self_, scale_factor);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::upsample_nearest3d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
  profiler::RecordFunction profiler("upsample_nearest3d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<UpsampleNearest3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::make_shared<UpsampleNearest3DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->scale_factor = scale_factor;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "upsample_nearest3d_backward", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("scale_factor"), scale_factor);
  }
  auto grad_input = as_variable(baseType->upsample_nearest3d_backward(grad_output_, self_, scale_factor));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::_sigmoid_out(Tensor & output, const Tensor & self) const {
  profiler::RecordFunction profiler("_sigmoid_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "_sigmoid_out", { output, self } );
  
  }
  Type::_sigmoid_out(output, self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::_sigmoid(const Tensor & self) const {
  profiler::RecordFunction profiler("_sigmoid");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "_sigmoid", { self } );
  
  }
  auto output = Type::_sigmoid(self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::_sigmoid_forward_out(Tensor & output, const Tensor & self) const {
  profiler::RecordFunction profiler("_sigmoid_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_sigmoid_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_sigmoid_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "_sigmoid_forward_out", { output, self } );
  
  }
  baseType->_sigmoid_forward_out(output_, self_);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::_sigmoid_forward(const Tensor & self) const {
  profiler::RecordFunction profiler("_sigmoid_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for _sigmoid_forward is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "_sigmoid_forward", { self } );
  
  }
  auto output = as_variable(baseType->_sigmoid_forward(self_));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
  profiler::RecordFunction profiler("_sigmoid_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& output_ = unpack(output, "output", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    throw_error_out_requires_grad("_sigmoid_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_sigmoid_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, output )) {
    trace_info = jit::tracer::preRecordTrace( "_sigmoid_backward_out", { grad_input, grad_output, output } );
  
  }
  baseType->_sigmoid_backward_out(grad_input_, grad_output_, output_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::_sigmoid_backward(const Tensor & grad_output, const Tensor & output) const {
  profiler::RecordFunction profiler("_sigmoid_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  std::shared_ptr<SigmoidBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    grad_fn = std::make_shared<SigmoidBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, output ));
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, output )) {
    trace_info = jit::tracer::preRecordTrace( "_sigmoid_backward", { grad_output, output } );
  
  }
  auto grad_input = as_variable(baseType->_sigmoid_backward(grad_output_, output_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::_tanh_out(Tensor & output, const Tensor & self) const {
  profiler::RecordFunction profiler("_tanh_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "_tanh_out", { output, self } );
  
  }
  Type::_tanh_out(output, self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::_tanh(const Tensor & self) const {
  profiler::RecordFunction profiler("_tanh");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "_tanh", { self } );
  
  }
  auto output = Type::_tanh(self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::_tanh_forward_out(Tensor & output, const Tensor & self) const {
  profiler::RecordFunction profiler("_tanh_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_tanh_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_tanh_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self )) {
    trace_info = jit::tracer::preRecordTrace( "_tanh_forward_out", { output, self } );
  
  }
  baseType->_tanh_forward_out(output_, self_);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::_tanh_forward(const Tensor & self) const {
  profiler::RecordFunction profiler("_tanh_forward");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<Error>("the derivative for _tanh_forward is not implemented");
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "_tanh_forward", { self } );
  
  }
  auto output = as_variable(baseType->_tanh_forward(self_));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::_tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
  profiler::RecordFunction profiler("_tanh_backward_out");
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& output_ = unpack(output, "output", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    throw_error_out_requires_grad("_tanh_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_tanh_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_output, output )) {
    trace_info = jit::tracer::preRecordTrace( "_tanh_backward_out", { grad_input, grad_output, output } );
  
  }
  baseType->_tanh_backward_out(grad_input_, grad_output_, output_);
  increment_version(grad_input);
  rebase_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input} );
  }
  return grad_input;
}
Tensor VariableType::_tanh_backward(const Tensor & grad_output, const Tensor & output) const {
  profiler::RecordFunction profiler("_tanh_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  std::shared_ptr<TanhBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    grad_fn = std::make_shared<TanhBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, output ));
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, output )) {
    trace_info = jit::tracer::preRecordTrace( "_tanh_backward", { grad_output, output } );
  
  }
  auto grad_input = as_variable(baseType->_tanh_backward(grad_output_, output_));
  set_history(grad_input, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input } );
  }
  return grad_input;
}
Tensor & VariableType::thnn_batch_norm_out(Tensor & output, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
  profiler::RecordFunction profiler("thnn_batch_norm_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, weight, bias, running_mean, running_var )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_batch_norm_out", { output, self, weight, bias, running_mean, running_var } );
    setattr(trace_info.n, jit::Symbol("training"), training);
    setattr(trace_info.n, jit::Symbol("momentum"), momentum);
    setattr(trace_info.n, jit::Symbol("eps"), eps);
  }
  Type::thnn_batch_norm_out(output, self, weight, bias, running_mean, running_var, training, momentum, eps);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::thnn_batch_norm(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
  profiler::RecordFunction profiler("thnn_batch_norm");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias, running_mean, running_var )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_batch_norm", { self, weight, bias, running_mean, running_var } );
    setattr(trace_info.n, jit::Symbol("training"), training);
    setattr(trace_info.n, jit::Symbol("momentum"), momentum);
    setattr(trace_info.n, jit::Symbol("eps"), eps);
  }
  auto output = Type::thnn_batch_norm(self, weight, bias, running_mean, running_var, training, momentum, eps);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_batch_norm_forward_out(Tensor & output, Tensor & save_mean, Tensor & save_std, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
  profiler::RecordFunction profiler("thnn_batch_norm_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& save_mean_ = unpack(save_mean, "save_mean", 1);
  auto& save_std_ = unpack(save_std, "save_std", 2);
  auto& self_ = unpack(self, "self", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 5);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 6);
  auto running_var_ = unpack_opt(running_var, "running_var", 7);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias, running_mean, running_var )) {
    throw_error_out_requires_grad("thnn_batch_norm_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_batch_norm_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, save_mean, save_std, self, weight, bias, running_mean, running_var )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_batch_norm_forward_out", { output, save_mean, save_std, self, weight, bias, running_mean, running_var } );
    setattr(trace_info.n, jit::Symbol("training"), training);
    setattr(trace_info.n, jit::Symbol("momentum"), momentum);
    setattr(trace_info.n, jit::Symbol("eps"), eps);
  }
  baseType->thnn_batch_norm_forward_out(output_, save_mean_, save_std_, self_, weight_, bias_, running_mean_, running_var_, training, momentum, eps);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, save_mean, save_std} );
  }
  return std::forward_as_tuple(output, save_mean, save_std);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_batch_norm_forward(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
  profiler::RecordFunction profiler("thnn_batch_norm_forward");
  auto& self_ = unpack(self, "self", 0);
  auto weight_ = unpack_opt(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<ThnnBatchNormBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::make_shared<ThnnBatchNormBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->training = training;
    grad_fn->eps = eps;
  }
  Tensor output;
  Tensor save_mean;
  Tensor save_std;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias, running_mean, running_var )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_batch_norm_forward", { self, weight, bias, running_mean, running_var } );
    setattr(trace_info.n, jit::Symbol("training"), training);
    setattr(trace_info.n, jit::Symbol("momentum"), momentum);
    setattr(trace_info.n, jit::Symbol("eps"), eps);
  }
  std::tie(output, save_mean, save_std) = as_variable(baseType->thnn_batch_norm_forward(self_, weight_, bias_, running_mean_, running_var_, training, momentum, eps));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, save_mean, save_std } );
  }
  if (grad_fn) {
    grad_fn->save_mean_ = SavedVariable(save_mean, true);
    grad_fn->save_std_ = SavedVariable(save_std, true);
  }
  return std::make_tuple(std::move(output), std::move(save_mean), std::move(save_std));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_batch_norm_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std) const {
  profiler::RecordFunction profiler("thnn_batch_norm_backward_out");
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto grad_bias_ = unpack_opt(grad_bias, "grad_bias", 2);
  auto& grad_output_ = unpack(grad_output, "grad_output", 3);
  auto& self_ = unpack(self, "self", 4);
  auto weight_ = unpack_opt(weight, "weight", 5);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 6);
  auto running_var_ = unpack_opt(running_var, "running_var", 7);
  auto save_mean_ = unpack_opt(save_mean, "save_mean", 10);
  auto save_std_ = unpack_opt(save_std, "save_std", 11);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, running_mean, running_var, save_mean, save_std )) {
    throw_error_out_requires_grad("thnn_batch_norm_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("thnn_batch_norm_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_weight, grad_bias, grad_output, self, weight, running_mean, running_var, save_mean, save_std )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_batch_norm_backward_out", { grad_input, grad_weight, grad_bias, grad_output, self, weight, running_mean, running_var, save_mean, save_std } );
    setattr(trace_info.n, jit::Symbol("training"), training);
    setattr(trace_info.n, jit::Symbol("eps"), eps);
  }
  baseType->thnn_batch_norm_backward_out(grad_input_, grad_weight_, grad_bias_, grad_output_, self_, weight_, running_mean_, running_var_, training, eps, save_mean_, save_std_);
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  rebase_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input, grad_weight, grad_bias} );
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_batch_norm_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("thnn_batch_norm_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto weight_ = unpack_opt(weight, "weight", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  auto save_mean_ = unpack_opt(save_mean, "save_mean", 7);
  auto save_std_ = unpack_opt(save_std, "save_std", 8);
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<ThnnBatchNormBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, save_mean, save_std )) {
    grad_fn = std::make_shared<ThnnBatchNormBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight, save_mean, save_std ));
    grad_fn->save_mean_ = SavedVariable(save_mean, false);
    grad_fn->save_std_ = SavedVariable(save_std, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->training = training;
    grad_fn->eps = eps;
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, weight, running_mean, running_var, save_mean, save_std )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_batch_norm_backward", { grad_output, self, weight, running_mean, running_var, save_mean, save_std } );
    setattr(trace_info.n, jit::Symbol("training"), training);
    setattr(trace_info.n, jit::Symbol("eps"), eps);
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->thnn_batch_norm_backward(grad_output_, self_, weight_, running_mean_, running_var_, training, eps, save_mean_, save_std_, output_mask));
  set_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input, grad_weight, grad_bias } );
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor & VariableType::thnn_conv_transpose2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_transpose2d_out", { output, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  Type::thnn_conv_transpose2d_out(output, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::thnn_conv_transpose2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose2d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_transpose2d", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  auto output = Type::thnn_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& columns_ = unpack(columns, "columns", 1);
  auto& ones_ = unpack(ones, "ones", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv_transpose2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv_transpose2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, columns, ones, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_transpose2d_forward_out", { output, columns, ones, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  baseType->thnn_conv_transpose2d_forward_out(output_, columns_, ones_, self_, weight_, kernel_size, bias_, stride, padding, output_padding, dilation);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, columns, ones} );
  }
  return std::forward_as_tuple(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_transpose2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose2d_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConvTranspose2DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::make_shared<ThnnConvTranspose2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->output_padding = output_padding;
    grad_fn->dilation = dilation;
  }
  Tensor output;
  Tensor columns;
  Tensor ones;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_transpose2d_forward", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  std::tie(output, columns, ones) = as_variable(baseType->thnn_conv_transpose2d_forward(self_, weight_, kernel_size, bias_, stride, padding, output_padding, dilation));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, columns, ones } );
  }
  if (grad_fn) {
    grad_fn->columns_ = SavedVariable(columns, true);
    grad_fn->ones_ = SavedVariable(ones, true);
  }
  return std::make_tuple(std::move(output), std::move(columns), std::move(ones));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
  profiler::RecordFunction profiler("thnn_conv_transpose2d_backward_out");
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto grad_bias_ = unpack_opt(grad_bias, "grad_bias", 2);
  auto& grad_output_ = unpack(grad_output, "grad_output", 3);
  auto& self_ = unpack(self, "self", 4);
  auto& weight_ = unpack(weight, "weight", 5);
  auto& columns_ = unpack(columns, "columns", 11);
  auto& ones_ = unpack(ones, "ones", 12);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, columns, ones )) {
    throw_error_out_requires_grad("thnn_conv_transpose2d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("thnn_conv_transpose2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_weight, grad_bias, grad_output, self, weight, columns, ones )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_transpose2d_backward_out", { grad_input, grad_weight, grad_bias, grad_output, self, weight, columns, ones } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  baseType->thnn_conv_transpose2d_backward_out(grad_input_, grad_weight_, grad_bias_, grad_output_, self_, weight_, kernel_size, stride, padding, output_padding, dilation, columns_, ones_);
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  rebase_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input, grad_weight, grad_bias} );
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("thnn_conv_transpose2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& columns_ = unpack(columns, "columns", 8);
  auto& ones_ = unpack(ones, "ones", 9);
  check_no_requires_grad(columns, "columns");
  check_no_requires_grad(ones, "ones");
  std::shared_ptr<ThnnConvTranspose2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::make_shared<ThnnConvTranspose2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->output_padding = output_padding;
    grad_fn->dilation = dilation;
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, weight, columns, ones )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_transpose2d_backward", { grad_output, self, weight, columns, ones } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->thnn_conv_transpose2d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, output_padding, dilation, columns_, ones_, output_mask));
  set_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input, grad_weight, grad_bias } );
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor & VariableType::thnn_conv_transpose3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose3d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_transpose3d_out", { output, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  Type::thnn_conv_transpose3d_out(output, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::thnn_conv_transpose3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose3d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_transpose3d", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  auto output = Type::thnn_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv_transpose3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose3d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& finput_ = unpack(finput, "finput", 1);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv_transpose3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv_transpose3d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, finput, fgrad_input, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_transpose3d_forward_out", { output, finput, fgrad_input, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  baseType->thnn_conv_transpose3d_forward_out(output_, finput_, fgrad_input_, self_, weight_, kernel_size, bias_, stride, padding, output_padding, dilation);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, finput, fgrad_input} );
  }
  return std::forward_as_tuple(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose3d_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConvTranspose3DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::make_shared<ThnnConvTranspose3DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->output_padding = output_padding;
    grad_fn->dilation = dilation;
  }
  Tensor output;
  Tensor finput;
  Tensor fgrad_input;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_transpose3d_forward", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  std::tie(output, finput, fgrad_input) = as_variable(baseType->thnn_conv_transpose3d_forward(self_, weight_, kernel_size, bias_, stride, padding, output_padding, dilation));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, finput, fgrad_input } );
  }
  if (grad_fn) {
    grad_fn->finput_ = SavedVariable(finput, true);
    grad_fn->fgrad_input_ = SavedVariable(fgrad_input, true);
  }
  return std::make_tuple(std::move(output), std::move(finput), std::move(fgrad_input));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const {
  profiler::RecordFunction profiler("thnn_conv_transpose3d_backward_out");
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto grad_bias_ = unpack_opt(grad_bias, "grad_bias", 2);
  auto& grad_output_ = unpack(grad_output, "grad_output", 3);
  auto& self_ = unpack(self, "self", 4);
  auto& weight_ = unpack(weight, "weight", 5);
  auto& finput_ = unpack(finput, "finput", 11);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 12);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, finput, fgrad_input )) {
    throw_error_out_requires_grad("thnn_conv_transpose3d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("thnn_conv_transpose3d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_transpose3d_backward_out", { grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  baseType->thnn_conv_transpose3d_backward_out(grad_input_, grad_weight_, grad_bias_, grad_output_, self_, weight_, kernel_size, stride, padding, output_padding, dilation, finput_, fgrad_input_);
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  rebase_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input, grad_weight, grad_bias} );
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("thnn_conv_transpose3d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 8);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 9);
  check_no_requires_grad(finput, "finput");
  check_no_requires_grad(fgrad_input, "fgrad_input");
  std::shared_ptr<ThnnConvTranspose3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::make_shared<ThnnConvTranspose3DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->output_padding = output_padding;
    grad_fn->dilation = dilation;
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, weight, finput, fgrad_input )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_transpose3d_backward", { grad_output, self, weight, finput, fgrad_input } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->thnn_conv_transpose3d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, output_padding, dilation, finput_, fgrad_input_, output_mask));
  set_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input, grad_weight, grad_bias } );
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor & VariableType::thnn_conv2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv2d_out", { output, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  Type::thnn_conv2d_out(output, self, weight, kernel_size, bias, stride, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::thnn_conv2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv2d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv2d", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = Type::thnn_conv2d(self, weight, kernel_size, bias, stride, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& finput_ = unpack(finput, "finput", 1);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, finput, fgrad_input, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv2d_forward_out", { output, finput, fgrad_input, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->thnn_conv2d_forward_out(output_, finput_, fgrad_input_, self_, weight_, kernel_size, bias_, stride, padding);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, finput, fgrad_input} );
  }
  return std::forward_as_tuple(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv2d_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConv2DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::make_shared<ThnnConv2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
  }
  Tensor output;
  Tensor finput;
  Tensor fgrad_input;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv2d_forward", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  std::tie(output, finput, fgrad_input) = as_variable(baseType->thnn_conv2d_forward(self_, weight_, kernel_size, bias_, stride, padding));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, finput, fgrad_input } );
  }
  if (grad_fn) {
    grad_fn->finput_ = SavedVariable(finput, true);
    grad_fn->fgrad_input_ = SavedVariable(fgrad_input, true);
  }
  return std::make_tuple(std::move(output), std::move(finput), std::move(fgrad_input));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
  profiler::RecordFunction profiler("thnn_conv2d_backward_out");
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto grad_bias_ = unpack_opt(grad_bias, "grad_bias", 2);
  auto& grad_output_ = unpack(grad_output, "grad_output", 3);
  auto& self_ = unpack(self, "self", 4);
  auto& weight_ = unpack(weight, "weight", 5);
  auto& finput_ = unpack(finput, "finput", 9);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 10);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, finput, fgrad_input )) {
    throw_error_out_requires_grad("thnn_conv2d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("thnn_conv2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv2d_backward_out", { grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->thnn_conv2d_backward_out(grad_input_, grad_weight_, grad_bias_, grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_);
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  rebase_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input, grad_weight, grad_bias} );
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("thnn_conv2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 6);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 7);
  check_no_requires_grad(finput, "finput");
  check_no_requires_grad(fgrad_input, "fgrad_input");
  std::shared_ptr<ThnnConv2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::make_shared<ThnnConv2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride;
    grad_fn->padding = padding;
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, weight, finput, fgrad_input )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv2d_backward", { grad_output, self, weight, finput, fgrad_input } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->thnn_conv2d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_, output_mask));
  set_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input, grad_weight, grad_bias } );
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor & VariableType::thnn_conv_depthwise2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_depthwise2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_depthwise2d_out", { output, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  Type::thnn_conv_depthwise2d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_depthwise2d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_depthwise2d", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  auto output = Type::thnn_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
Tensor & VariableType::thnn_conv_depthwise2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_depthwise2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto bias_ = unpack_opt(bias, "bias", 4);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv_depthwise2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv_depthwise2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_depthwise2d_forward_out", { output, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  baseType->thnn_conv_depthwise2d_forward_out(output_, self_, weight_, kernel_size, bias_, stride, padding, dilation);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_depthwise2d_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConvDepthwise2DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::make_shared<ThnnConvDepthwise2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->dilation = dilation;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_depthwise2d_forward", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  auto output = as_variable(baseType->thnn_conv_depthwise2d_forward(self_, weight_, kernel_size, bias_, stride, padding, dilation));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &> VariableType::thnn_conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_depthwise2d_backward_out");
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto& grad_output_ = unpack(grad_output, "grad_output", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    throw_error_out_requires_grad("thnn_conv_depthwise2d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight )) {
    throw_error_out_requires_grad("thnn_conv_depthwise2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_weight, grad_output, self, weight )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_depthwise2d_backward_out", { grad_input, grad_weight, grad_output, self, weight } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  baseType->thnn_conv_depthwise2d_backward_out(grad_input_, grad_weight_, grad_output_, self_, weight_, kernel_size, stride, padding, dilation);
  increment_version(grad_input);
  increment_version(grad_weight);
  rebase_history({ grad_input, grad_weight }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input, grad_weight} );
  }
  return std::forward_as_tuple(grad_input, grad_weight);
}
std::tuple<Tensor,Tensor> VariableType::thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, std::array<bool,2> output_mask) const {
  profiler::RecordFunction profiler("thnn_conv_depthwise2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<ThnnConvDepthwise2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::make_shared<ThnnConvDepthwise2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_argsize_1 = self.size(1);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->dilation = dilation;
  }
  Tensor grad_input;
  Tensor grad_weight;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, weight )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_depthwise2d_backward", { grad_output, self, weight } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(grad_input, grad_weight) = as_variable(baseType->thnn_conv_depthwise2d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, dilation, output_mask));
  set_history({ grad_input, grad_weight }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input, grad_weight } );
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight));
}
Tensor & VariableType::thnn_conv3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv3d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv3d_out", { output, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  Type::thnn_conv3d_out(output, self, weight, kernel_size, bias, stride, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::thnn_conv3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv3d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv3d", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  auto output = Type::thnn_conv3d(self, weight, kernel_size, bias, stride, padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv3d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& finput_ = unpack(finput, "finput", 1);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv3d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, finput, fgrad_input, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv3d_forward_out", { output, finput, fgrad_input, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->thnn_conv3d_forward_out(output_, finput_, fgrad_input_, self_, weight_, kernel_size, bias_, stride, padding);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, finput, fgrad_input} );
  }
  return std::forward_as_tuple(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv3d_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConv3DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::make_shared<ThnnConv3DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
  }
  Tensor output;
  Tensor finput;
  Tensor fgrad_input;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv3d_forward", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  std::tie(output, finput, fgrad_input) = as_variable(baseType->thnn_conv3d_forward(self_, weight_, kernel_size, bias_, stride, padding));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, finput, fgrad_input } );
  }
  if (grad_fn) {
    grad_fn->finput_ = SavedVariable(finput, true);
    grad_fn->fgrad_input_ = SavedVariable(fgrad_input, true);
  }
  return std::make_tuple(std::move(output), std::move(finput), std::move(fgrad_input));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
  profiler::RecordFunction profiler("thnn_conv3d_backward_out");
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto grad_bias_ = unpack_opt(grad_bias, "grad_bias", 2);
  auto& grad_output_ = unpack(grad_output, "grad_output", 3);
  auto& self_ = unpack(self, "self", 4);
  auto& weight_ = unpack(weight, "weight", 5);
  auto& finput_ = unpack(finput, "finput", 9);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 10);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, finput, fgrad_input )) {
    throw_error_out_requires_grad("thnn_conv3d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("thnn_conv3d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv3d_backward_out", { grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
  }
  baseType->thnn_conv3d_backward_out(grad_input_, grad_weight_, grad_bias_, grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_);
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  rebase_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input, grad_weight, grad_bias} );
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("thnn_conv3d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 6);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 7);
  check_no_requires_grad(finput, "finput");
  check_no_requires_grad(fgrad_input, "fgrad_input");
  std::shared_ptr<ThnnConv3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::make_shared<ThnnConv3DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride;
    grad_fn->padding = padding;
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, weight, finput, fgrad_input )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv3d_backward", { grad_output, self, weight, finput, fgrad_input } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->thnn_conv3d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_, output_mask));
  set_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input, grad_weight, grad_bias } );
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor & VariableType::thnn_conv_dilated2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated2d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_dilated2d_out", { output, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  Type::thnn_conv_dilated2d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::thnn_conv_dilated2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated2d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_dilated2d", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  auto output = Type::thnn_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv_dilated2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated2d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& columns_ = unpack(columns, "columns", 1);
  auto& ones_ = unpack(ones, "ones", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv_dilated2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv_dilated2d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, columns, ones, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_dilated2d_forward_out", { output, columns, ones, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  baseType->thnn_conv_dilated2d_forward_out(output_, columns_, ones_, self_, weight_, kernel_size, bias_, stride, padding, dilation);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, columns, ones} );
  }
  return std::forward_as_tuple(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated2d_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConvDilated2DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::make_shared<ThnnConvDilated2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->dilation = dilation;
  }
  Tensor output;
  Tensor columns;
  Tensor ones;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_dilated2d_forward", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  std::tie(output, columns, ones) = as_variable(baseType->thnn_conv_dilated2d_forward(self_, weight_, kernel_size, bias_, stride, padding, dilation));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, columns, ones } );
  }
  if (grad_fn) {
    grad_fn->columns_ = SavedVariable(columns, true);
    grad_fn->ones_ = SavedVariable(ones, true);
  }
  return std::make_tuple(std::move(output), std::move(columns), std::move(ones));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
  profiler::RecordFunction profiler("thnn_conv_dilated2d_backward_out");
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto grad_bias_ = unpack_opt(grad_bias, "grad_bias", 2);
  auto& grad_output_ = unpack(grad_output, "grad_output", 3);
  auto& self_ = unpack(self, "self", 4);
  auto& weight_ = unpack(weight, "weight", 5);
  auto& columns_ = unpack(columns, "columns", 10);
  auto& ones_ = unpack(ones, "ones", 11);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, columns, ones )) {
    throw_error_out_requires_grad("thnn_conv_dilated2d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("thnn_conv_dilated2d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_weight, grad_bias, grad_output, self, weight, columns, ones )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_dilated2d_backward_out", { grad_input, grad_weight, grad_bias, grad_output, self, weight, columns, ones } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  baseType->thnn_conv_dilated2d_backward_out(grad_input_, grad_weight_, grad_bias_, grad_output_, self_, weight_, kernel_size, stride, padding, dilation, columns_, ones_);
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  rebase_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input, grad_weight, grad_bias} );
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("thnn_conv_dilated2d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& columns_ = unpack(columns, "columns", 7);
  auto& ones_ = unpack(ones, "ones", 8);
  check_no_requires_grad(columns, "columns");
  check_no_requires_grad(ones, "ones");
  std::shared_ptr<ThnnConvDilated2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::make_shared<ThnnConvDilated2DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->dilation = dilation;
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, weight, columns, ones )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_dilated2d_backward", { grad_output, self, weight, columns, ones } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->thnn_conv_dilated2d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, dilation, columns_, ones_, output_mask));
  set_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input, grad_weight, grad_bias } );
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor & VariableType::thnn_conv_dilated3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated3d_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_dilated3d_out", { output, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  Type::thnn_conv_dilated3d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output} );
  }
  return output;
}
Tensor VariableType::thnn_conv_dilated3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated3d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_dilated3d", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  auto output = Type::thnn_conv_dilated3d(self, weight, kernel_size, bias, stride, padding, dilation);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated3d_forward_out");
  auto& output_ = unpack(output, "output", 0);
  auto& columns_ = unpack(columns, "columns", 1);
  auto& ones_ = unpack(ones, "ones", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv_dilated3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv_dilated3d_forward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( output, columns, ones, self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_dilated3d_forward_out", { output, columns, ones, self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  baseType->thnn_conv_dilated3d_forward_out(output_, columns_, ones_, self_, weight_, kernel_size, bias_, stride, padding, dilation);
  increment_version(output);
  rebase_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {output, columns, ones} );
  }
  return std::forward_as_tuple(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated3d_forward");
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConvDilated3DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::make_shared<ThnnConvDilated3DBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size;
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->dilation = dilation;
  }
  Tensor output;
  Tensor columns;
  Tensor ones;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_dilated3d_forward", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  std::tie(output, columns, ones) = as_variable(baseType->thnn_conv_dilated3d_forward(self_, weight_, kernel_size, bias_, stride, padding, dilation));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output, columns, ones } );
  }
  if (grad_fn) {
    grad_fn->columns_ = SavedVariable(columns, true);
    grad_fn->ones_ = SavedVariable(ones, true);
  }
  return std::make_tuple(std::move(output), std::move(columns), std::move(ones));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
  profiler::RecordFunction profiler("thnn_conv_dilated3d_backward_out");
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto grad_bias_ = unpack_opt(grad_bias, "grad_bias", 2);
  auto& grad_output_ = unpack(grad_output, "grad_output", 3);
  auto& self_ = unpack(self, "self", 4);
  auto& weight_ = unpack(weight, "weight", 5);
  auto& columns_ = unpack(columns, "columns", 10);
  auto& ones_ = unpack(ones, "ones", 11);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, columns, ones )) {
    throw_error_out_requires_grad("thnn_conv_dilated3d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("thnn_conv_dilated3d_backward");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_input, grad_weight, grad_bias, grad_output, self, weight, columns, ones )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_dilated3d_backward_out", { grad_input, grad_weight, grad_bias, grad_output, self, weight, columns, ones } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
  }
  baseType->thnn_conv_dilated3d_backward_out(grad_input_, grad_weight_, grad_bias_, grad_output_, self_, weight_, kernel_size, stride, padding, dilation, columns_, ones_);
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  rebase_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {grad_input, grad_weight, grad_bias} );
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("thnn_conv_dilated3d_backward");
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& columns_ = unpack(columns, "columns", 7);
  auto& ones_ = unpack(ones, "ones", 8);
  check_no_requires_grad(columns, "columns");
  check_no_requires_grad(ones, "ones");
  std::shared_ptr<ThnnConvDilated3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::make_shared<ThnnConvDilated3DBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride;
    grad_fn->padding = padding;
    grad_fn->dilation = dilation;
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self, weight, columns, ones )) {
    trace_info = jit::tracer::preRecordTrace( "thnn_conv_dilated3d_backward", { grad_output, self, weight, columns, ones } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->thnn_conv_dilated3d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, dilation, columns_, ones_, output_mask));
  set_history({ grad_input, grad_weight, grad_bias }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_input, grad_weight, grad_bias } );
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor VariableType::adaptive_avg_pool1d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_avg_pool1d");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_avg_pool1d", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  auto result = Type::adaptive_avg_pool1d(self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::adaptive_max_pool1d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool1d");
  Tensor result0;
  Tensor result1;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "adaptive_max_pool1d", { self } );
    setattr(trace_info.n, jit::Symbol("output_size"), output_size);
  }
  std::tie(result0, result1) = Type::adaptive_max_pool1d(self, output_size);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result0, result1 } );
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
bool VariableType::allclose(const Tensor & self, const Tensor & other, double rtol, double atol) const {
  profiler::RecordFunction profiler("allclose");
  auto result = Type::allclose(self, other, rtol, atol);
  return result;
}
Tensor VariableType::addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmv");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mat, vec )) {
    trace_info = jit::tracer::preRecordTrace( "addmv", { self, mat, vec } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = Type::addmv(self, mat, vec, beta, alpha);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmv_");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mat, vec )) {
    trace_info = jit::tracer::preRecordTrace( "addmv", { self, mat, vec } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  Type::addmv_(self, mat, vec, beta, alpha);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmv_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, mat, vec )) {
    trace_info = jit::tracer::preRecordTrace( "addmv_out", { result, self, mat, vec } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  Type::addmv_out(result, self, mat, vec, beta, alpha);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addr");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, vec1, vec2 )) {
    trace_info = jit::tracer::preRecordTrace( "addr", { self, vec1, vec2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = Type::addr(self, vec1, vec2, beta, alpha);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addr_");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, vec1, vec2 )) {
    trace_info = jit::tracer::preRecordTrace( "addr", { self, vec1, vec2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  Type::addr_(self, vec1, vec2, beta, alpha);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addr_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, vec1, vec2 )) {
    trace_info = jit::tracer::preRecordTrace( "addr_out", { result, self, vec1, vec2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  Type::addr_out(result, self, vec1, vec2, beta, alpha);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) const {
  profiler::RecordFunction profiler("batch_norm");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( input, weight, bias, running_mean, running_var )) {
    trace_info = jit::tracer::preRecordTrace( "batch_norm", { input, weight, bias, running_mean, running_var } );
    setattr(trace_info.n, jit::Symbol("training"), training);
    setattr(trace_info.n, jit::Symbol("momentum"), momentum);
    setattr(trace_info.n, jit::Symbol("eps"), eps);
    setattr(trace_info.n, jit::Symbol("cudnn_enabled"), cudnn_enabled);
  }
  auto result = Type::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::bernoulli_(Tensor & self, const Tensor & p, Generator * generator) const {
  profiler::RecordFunction profiler("bernoulli_");
  Type::bernoulli_(self, p, generator);
  return self;
}
Tensor & VariableType::bernoulli_(Tensor & self, double p, Generator * generator) const {
  profiler::RecordFunction profiler("bernoulli_");
  Type::bernoulli_(self, p, generator);
  return self;
}
Tensor VariableType::cat(TensorList tensors, int64_t dim) const {
  profiler::RecordFunction profiler("cat");
  auto tensors_ = unpack(tensors, "tensors", 0);
  std::shared_ptr<CatBackward> grad_fn;
  if (compute_requires_grad( tensors )) {
    grad_fn = std::make_shared<CatBackward>();
    grad_fn->set_next_edges(collect_next_edges( tensors ));
    grad_fn->tensors_sizes_dim = to_arg_sizes(tensors, dim);
    grad_fn->dim = dim;
    grad_fn->tensors_size_ = tensors.size();
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( tensors )) {
    trace_info = jit::tracer::preRecordTrace( "cat", flatten( tensors ) );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto result = as_variable(baseType->cat(tensors_, dim));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::cat_out(Tensor & result, TensorList tensors, int64_t dim) const {
  profiler::RecordFunction profiler("cat_out");
  auto& result_ = unpack(result, "result", 0);
  auto tensors_ = unpack(tensors, "tensors", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( tensors )) {
    throw_error_out_requires_grad("cat");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("cat");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, tensors )) {
    trace_info = jit::tracer::preRecordTrace( "cat_out", flatten( result, tensors ) );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->cat_out(result_, tensors_, dim);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
std::vector<Tensor> VariableType::chunk(const Tensor & self, int64_t chunks, int64_t dim) const {
  profiler::RecordFunction profiler("chunk");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "chunk", { self } );
    setattr(trace_info.n, jit::Symbol("chunks"), chunks);
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto result = Type::chunk(self, chunks, dim);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  flatten(result) );
  }
  return result;
}
bool VariableType::cudnn_is_acceptable(const Tensor & self) const {
  profiler::RecordFunction profiler("cudnn_is_acceptable");
  auto result = Type::cudnn_is_acceptable(self);
  return result;
}
Tensor VariableType::convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups) const {
  profiler::RecordFunction profiler("convolution");
  auto result = Type::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  return result;
}
Tensor VariableType::_convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) const {
  profiler::RecordFunction profiler("_convolution");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( input, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "_convolution", { input, weight, bias } );
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("transposed"), transposed);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("groups"), groups);
    setattr(trace_info.n, jit::Symbol("benchmark"), benchmark);
    setattr(trace_info.n, jit::Symbol("deterministic"), deterministic);
    setattr(trace_info.n, jit::Symbol("cudnn_enabled"), cudnn_enabled);
  }
  auto result = Type::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::_convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding) const {
  profiler::RecordFunction profiler("_convolution_nogroup");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( input, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "_convolution_nogroup", { input, weight, bias } );
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("transposed"), transposed);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
  }
  auto result = Type::_convolution_nogroup(input, weight, bias, stride, padding, dilation, transposed, output_padding);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_convolution_double_backward(const Tensor & ggI, const Tensor & ggW, const Tensor & ggb, const Tensor & gO, const Tensor & weight, const Tensor & self, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("_convolution_double_backward");
  Tensor result0;
  Tensor result1;
  Tensor result2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( ggI, ggW, ggb, gO, weight, self )) {
    trace_info = jit::tracer::preRecordTrace( "_convolution_double_backward", { ggI, ggW, ggb, gO, weight, self } );
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("transposed"), transposed);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("groups"), groups);
    setattr(trace_info.n, jit::Symbol("benchmark"), benchmark);
    setattr(trace_info.n, jit::Symbol("deterministic"), deterministic);
    setattr(trace_info.n, jit::Symbol("cudnn_enabled"), cudnn_enabled);
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(result0, result1, result2) = Type::_convolution_double_backward(ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result0, result1, result2 } );
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) const {
  profiler::RecordFunction profiler("conv1d");
  auto result = Type::conv1d(input, weight, bias, stride, padding, dilation, groups);
  return result;
}
Tensor VariableType::conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) const {
  profiler::RecordFunction profiler("conv2d");
  auto result = Type::conv2d(input, weight, bias, stride, padding, dilation, groups);
  return result;
}
Tensor VariableType::conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) const {
  profiler::RecordFunction profiler("conv3d");
  auto result = Type::conv3d(input, weight, bias, stride, padding, dilation, groups);
  return result;
}
Tensor VariableType::conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) const {
  profiler::RecordFunction profiler("conv_tbc");
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto& bias_ = unpack(bias, "bias", 2);
  std::shared_ptr<ConvTbcBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::make_shared<ConvTbcBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->bias_ = SavedVariable(bias, false);
    grad_fn->pad = pad;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "conv_tbc", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("pad"), pad);
  }
  auto result = as_variable(baseType->conv_tbc(self_, weight_, bias_, pad));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::conv_tbc_backward(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad) const {
  profiler::RecordFunction profiler("conv_tbc_backward");
  Tensor result0;
  Tensor result1;
  Tensor result2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, input, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "conv_tbc_backward", { self, input, weight, bias } );
    setattr(trace_info.n, jit::Symbol("pad"), pad);
  }
  std::tie(result0, result1, result2) = Type::conv_tbc_backward(self, input, weight, bias, pad);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result0, result1, result2 } );
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) const {
  profiler::RecordFunction profiler("conv_transpose1d");
  auto result = Type::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
Tensor VariableType::conv_transpose2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) const {
  profiler::RecordFunction profiler("conv_transpose2d");
  auto result = Type::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
Tensor VariableType::conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) const {
  profiler::RecordFunction profiler("conv_transpose3d");
  auto result = Type::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
Tensor VariableType::cudnn_affine_grid_generator(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) const {
  profiler::RecordFunction profiler("cudnn_affine_grid_generator");
  auto& theta_ = unpack(theta, "theta", 0);
  std::shared_ptr<CudnnAffineGridGeneratorBackward> grad_fn;
  if (compute_requires_grad( theta )) {
    grad_fn = std::make_shared<CudnnAffineGridGeneratorBackward>();
    grad_fn->set_next_edges(collect_next_edges( theta ));
    grad_fn->N = N;
    grad_fn->C = C;
    grad_fn->H = H;
    grad_fn->W = W;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( theta )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_affine_grid_generator", { theta } );
    setattr(trace_info.n, jit::Symbol("N"), N);
    setattr(trace_info.n, jit::Symbol("C"), C);
    setattr(trace_info.n, jit::Symbol("H"), H);
    setattr(trace_info.n, jit::Symbol("W"), W);
  }
  auto grid = as_variable(baseType->cudnn_affine_grid_generator(theta_, N, C, H, W));
  set_history(grid, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grid } );
  }
  return grid;
}
Tensor VariableType::cudnn_affine_grid_generator_backward(const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) const {
  profiler::RecordFunction profiler("cudnn_affine_grid_generator_backward");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_affine_grid_generator_backward", { grad } );
    setattr(trace_info.n, jit::Symbol("N"), N);
    setattr(trace_info.n, jit::Symbol("C"), C);
    setattr(trace_info.n, jit::Symbol("H"), H);
    setattr(trace_info.n, jit::Symbol("W"), W);
  }
  auto grad_theta = Type::cudnn_affine_grid_generator_backward(grad, N, C, H, W);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_theta } );
  }
  return grad_theta;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) const {
  profiler::RecordFunction profiler("cudnn_batch_norm");
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<CudnnBatchNormBackward> grad_fn;
  if (compute_requires_grad( input, weight, bias )) {
    grad_fn = std::make_shared<CudnnBatchNormBackward>();
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->training = training;
    grad_fn->epsilon = epsilon;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( input, weight, bias, running_mean, running_var )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_batch_norm", { input, weight, bias, running_mean, running_var } );
    setattr(trace_info.n, jit::Symbol("training"), training);
    setattr(trace_info.n, jit::Symbol("exponential_average_factor"), exponential_average_factor);
    setattr(trace_info.n, jit::Symbol("epsilon"), epsilon);
  }
  std::tie(result0, result1, result2) = as_variable(baseType->cudnn_batch_norm(input_, weight_, bias_, running_mean_, running_var_, training, exponential_average_factor, epsilon));
  set_history(result0, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result0, result1, result2 } );
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor,Tensor> VariableType::cudnn_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon) const {
  profiler::RecordFunction profiler("cudnn_batch_norm_backward");
  auto& input_ = unpack(input, "input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  auto save_mean_ = unpack_opt(save_mean, "save_mean", 5);
  auto save_var_ = unpack_opt(save_var, "save_var", 6);
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<CudnnBatchNormBackwardBackward> grad_fn;
  if (compute_requires_grad( input, grad_output, weight, save_mean, save_var )) {
    grad_fn = std::make_shared<CudnnBatchNormBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( input, grad_output, weight, save_mean, save_var ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->save_mean_ = SavedVariable(save_mean, false);
    grad_fn->save_var_ = SavedVariable(save_var, false);
    grad_fn->epsilon = epsilon;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( input, grad_output, weight, running_mean, running_var, save_mean, save_var )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_batch_norm_backward", { input, grad_output, weight, running_mean, running_var, save_mean, save_var } );
    setattr(trace_info.n, jit::Symbol("epsilon"), epsilon);
  }
  std::tie(result0, result1, result2) = as_variable(baseType->cudnn_batch_norm_backward(input_, grad_output_, weight_, running_mean_, running_var_, save_mean_, save_var_, epsilon));
  set_history({ result0, result1, result2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result0, result1, result2 } );
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::cudnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("cudnn_convolution");
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<CudnnConvolutionBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::make_shared<CudnnConvolutionBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding;
    grad_fn->stride = stride;
    grad_fn->dilation = dilation;
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_convolution", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("groups"), groups);
    setattr(trace_info.n, jit::Symbol("benchmark"), benchmark);
    setattr(trace_info.n, jit::Symbol("deterministic"), deterministic);
  }
  auto result = as_variable(baseType->cudnn_convolution(self_, weight_, bias_, padding, stride, dilation, groups, benchmark, deterministic));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::cudnn_convolution_backward_input(IntList self_size, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("cudnn_convolution_backward_input");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, weight )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_convolution_backward_input", { grad_output, weight } );
    setattr(trace_info.n, jit::Symbol("self_size"), self_size);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("groups"), groups);
    setattr(trace_info.n, jit::Symbol("benchmark"), benchmark);
    setattr(trace_info.n, jit::Symbol("deterministic"), deterministic);
  }
  auto result = Type::cudnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::cudnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("cudnn_convolution_backward");
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<CudnnConvolutionBackwardBackward> grad_fn;
  if (compute_requires_grad( self, grad_output, weight )) {
    grad_fn = std::make_shared<CudnnConvolutionBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, grad_output, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding;
    grad_fn->stride = stride;
    grad_fn->dilation = dilation;
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, grad_output, weight )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_convolution_backward", { self, grad_output, weight } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("groups"), groups);
    setattr(trace_info.n, jit::Symbol("benchmark"), benchmark);
    setattr(trace_info.n, jit::Symbol("deterministic"), deterministic);
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(result0, result1, result2) = as_variable(baseType->cudnn_convolution_backward(self_, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic, output_mask));
  set_history({ result0, result1, result2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result0, result1, result2 } );
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::cudnn_convolution_backward_bias(const Tensor & grad_output) const {
  profiler::RecordFunction profiler("cudnn_convolution_backward_bias");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_convolution_backward_bias", { grad_output } );
  
  }
  auto result = Type::cudnn_convolution_backward_bias(grad_output);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::cudnn_convolution_backward_weight(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("cudnn_convolution_backward_weight");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_convolution_backward_weight", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("weight_size"), weight_size);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("groups"), groups);
    setattr(trace_info.n, jit::Symbol("benchmark"), benchmark);
    setattr(trace_info.n, jit::Symbol("deterministic"), deterministic);
  }
  auto result = Type::cudnn_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("cudnn_convolution_transpose");
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<CudnnConvolutionTransposeBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::make_shared<CudnnConvolutionTransposeBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding;
    grad_fn->output_padding = output_padding;
    grad_fn->stride = stride;
    grad_fn->dilation = dilation;
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_convolution_transpose", { self, weight, bias } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("groups"), groups);
    setattr(trace_info.n, jit::Symbol("benchmark"), benchmark);
    setattr(trace_info.n, jit::Symbol("deterministic"), deterministic);
  }
  auto result = as_variable(baseType->cudnn_convolution_transpose(self_, weight_, bias_, padding, output_padding, stride, dilation, groups, benchmark, deterministic));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::cudnn_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("cudnn_convolution_transpose_backward");
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<CudnnConvolutionTransposeBackwardBackward> grad_fn;
  if (compute_requires_grad( self, grad_output, weight )) {
    grad_fn = std::make_shared<CudnnConvolutionTransposeBackwardBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, grad_output, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding;
    grad_fn->output_padding = output_padding;
    grad_fn->stride = stride;
    grad_fn->dilation = dilation;
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, grad_output, weight )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_convolution_transpose_backward", { self, grad_output, weight } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("output_padding"), output_padding);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("groups"), groups);
    setattr(trace_info.n, jit::Symbol("benchmark"), benchmark);
    setattr(trace_info.n, jit::Symbol("deterministic"), deterministic);
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(result0, result1, result2) = as_variable(baseType->cudnn_convolution_transpose_backward(self_, grad_output_, weight_, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask));
  set_history({ result0, result1, result2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result0, result1, result2 } );
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::cudnn_convolution_transpose_backward_bias(const Tensor & grad_output) const {
  profiler::RecordFunction profiler("cudnn_convolution_transpose_backward_bias");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_convolution_transpose_backward_bias", { grad_output } );
  
  }
  auto result = Type::cudnn_convolution_transpose_backward_bias(grad_output);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::cudnn_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("cudnn_convolution_transpose_backward_input");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, weight )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_convolution_transpose_backward_input", { grad_output, weight } );
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("groups"), groups);
    setattr(trace_info.n, jit::Symbol("benchmark"), benchmark);
    setattr(trace_info.n, jit::Symbol("deterministic"), deterministic);
  }
  auto result = Type::cudnn_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::cudnn_convolution_transpose_backward_weight(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("cudnn_convolution_transpose_backward_weight");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad_output, self )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_convolution_transpose_backward_weight", { grad_output, self } );
    setattr(trace_info.n, jit::Symbol("weight_size"), weight_size);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("groups"), groups);
    setattr(trace_info.n, jit::Symbol("benchmark"), benchmark);
    setattr(trace_info.n, jit::Symbol("deterministic"), deterministic);
  }
  auto result = Type::cudnn_convolution_transpose_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::cudnn_grid_sampler(const Tensor & self, const Tensor & grid) const {
  profiler::RecordFunction profiler("cudnn_grid_sampler");
  auto& self_ = unpack(self, "self", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  std::shared_ptr<CudnnGridSamplerBackward> grad_fn;
  if (compute_requires_grad( self, grid )) {
    grad_fn = std::make_shared<CudnnGridSamplerBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, grid ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grid_ = SavedVariable(grid, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, grid )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_grid_sampler", { self, grid } );
  
  }
  auto output = as_variable(baseType->cudnn_grid_sampler(self_, grid_));
  set_history(output, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { output } );
  }
  return output;
}
std::tuple<Tensor,Tensor> VariableType::cudnn_grid_sampler_backward(const Tensor & self, const Tensor & grid, const Tensor & grad_output) const {
  profiler::RecordFunction profiler("cudnn_grid_sampler_backward");
  Tensor grad_self;
  Tensor grad_grid;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, grid, grad_output )) {
    trace_info = jit::tracer::preRecordTrace( "cudnn_grid_sampler_backward", { self, grid, grad_output } );
  
  }
  std::tie(grad_self, grad_grid) = Type::cudnn_grid_sampler_backward(self, grid, grad_output);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { grad_self, grad_grid } );
  }
  return std::make_tuple(std::move(grad_self), std::move(grad_grid));
}
Tensor VariableType::det(const Tensor & self) const {
  profiler::RecordFunction profiler("det");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "det", { self } );
  
  }
  auto result = Type::det(self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> VariableType::_det_with_svd(const Tensor & self) const {
  profiler::RecordFunction profiler("_det_with_svd");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DetWithSvdBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<DetWithSvdBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "_det_with_svd", { self } );
  
  }
  std::tie(result0, result1, result2, result3) = as_variable(baseType->_det_with_svd(self_));
  set_history({ result0, result1, result2, result3 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result0, result1, result2, result3 } );
  }
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
    grad_fn->result3_ = SavedVariable(result3, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
Tensor VariableType::dot(const Tensor & self, const Tensor & tensor) const {
  profiler::RecordFunction profiler("dot");
  auto& self_ = unpack(self, "self", 0);
  auto& tensor_ = unpack(tensor, "tensor", 1);
  std::shared_ptr<DotBackward> grad_fn;
  if (compute_requires_grad( self, tensor )) {
    grad_fn = std::make_shared<DotBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, tensor ));
    grad_fn->tensor_ = SavedVariable(tensor, false);
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, tensor )) {
    trace_info = jit::tracer::preRecordTrace( "dot", { self, tensor } );
  
  }
  auto result = as_variable(baseType->dot(self_, tensor_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) const {
  profiler::RecordFunction profiler("embedding");
  auto& weight_ = unpack(weight, "weight", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<EmbeddingBackward> grad_fn;
  if (compute_requires_grad( weight )) {
    grad_fn = std::make_shared<EmbeddingBackward>();
    grad_fn->set_next_edges(collect_next_edges( weight ));
    grad_fn->weight_argsize_0 = weight.size(0);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->padding_idx = padding_idx;
    grad_fn->scale_grad_by_freq = scale_grad_by_freq;
    grad_fn->sparse = sparse;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( weight, indices )) {
    trace_info = jit::tracer::preRecordTrace( "embedding", { weight, indices } );
    setattr(trace_info.n, jit::Symbol("padding_idx"), padding_idx);
    setattr(trace_info.n, jit::Symbol("scale_grad_by_freq"), scale_grad_by_freq);
    setattr(trace_info.n, jit::Symbol("sparse"), sparse);
  }
  auto result = as_variable(baseType->embedding(weight_, indices_, padding_idx, scale_grad_by_freq, sparse));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::embedding_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) const {
  profiler::RecordFunction profiler("embedding_backward");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad, indices )) {
    trace_info = jit::tracer::preRecordTrace( "embedding_backward", { grad, indices } );
    setattr(trace_info.n, jit::Symbol("num_weights"), num_weights);
    setattr(trace_info.n, jit::Symbol("padding_idx"), padding_idx);
    setattr(trace_info.n, jit::Symbol("scale_grad_by_freq"), scale_grad_by_freq);
    setattr(trace_info.n, jit::Symbol("sparse"), sparse);
  }
  auto result = Type::embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
  profiler::RecordFunction profiler("embedding_dense_backward");
  auto& grad_ = unpack(grad, "grad", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( grad )) {
    grad_fn = std::make_shared<Error>("the derivative for embedding_dense_backward is not implemented");
    grad_fn->set_next_edges(collect_next_edges( grad ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad, indices )) {
    trace_info = jit::tracer::preRecordTrace( "embedding_dense_backward", { grad, indices } );
    setattr(trace_info.n, jit::Symbol("num_weights"), num_weights);
    setattr(trace_info.n, jit::Symbol("padding_idx"), padding_idx);
    setattr(trace_info.n, jit::Symbol("scale_grad_by_freq"), scale_grad_by_freq);
  }
  auto result = as_variable(baseType->embedding_dense_backward(grad_, indices_, num_weights, padding_idx, scale_grad_by_freq));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
  profiler::RecordFunction profiler("embedding_renorm_");
  auto& self_ = unpack(self, "self", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  check_inplace(self);
  std::shared_ptr<EmbeddingRenormBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<EmbeddingRenormBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "embedding_renorm", { self, indices } );
    setattr(trace_info.n, jit::Symbol("max_norm"), max_norm);
    setattr(trace_info.n, jit::Symbol("norm_type"), norm_type);
  }
  baseType->embedding_renorm_(self_, indices_, max_norm, norm_type);
  increment_version(self);
  rebase_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor VariableType::embedding_sparse_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
  profiler::RecordFunction profiler("embedding_sparse_backward");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad, indices )) {
    trace_info = jit::tracer::preRecordTrace( "embedding_sparse_backward", { grad, indices } );
    setattr(trace_info.n, jit::Symbol("num_weights"), num_weights);
    setattr(trace_info.n, jit::Symbol("padding_idx"), padding_idx);
    setattr(trace_info.n, jit::Symbol("scale_grad_by_freq"), scale_grad_by_freq);
  }
  auto result = Type::embedding_sparse_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::empty_like(const Tensor & self) const {
  profiler::RecordFunction profiler("empty_like");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "empty_like", { self } );
  
  }
  auto result = Type::empty_like(self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::empty_like(const Tensor & self, const Type & dtype) const {
  profiler::RecordFunction profiler("empty_like");
  auto result = Type::empty_like(self, dtype);
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
  profiler::RecordFunction profiler("embedding_bag");
  auto& weight_ = unpack(weight, "weight", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& offsets_ = unpack(offsets, "offsets", 2);
  std::shared_ptr<EmbeddingBagBackward> grad_fn;
  if (compute_requires_grad( weight )) {
    grad_fn = std::make_shared<EmbeddingBagBackward>();
    grad_fn->set_next_edges(collect_next_edges( weight ));
    grad_fn->weight_argsize_0 = weight.size(0);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->offsets_ = SavedVariable(offsets, false);
    grad_fn->scale_grad_by_freq = scale_grad_by_freq;
    grad_fn->mode = mode;
    grad_fn->sparse = sparse;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( weight, indices, offsets )) {
    trace_info = jit::tracer::preRecordTrace( "embedding_bag", { weight, indices, offsets } );
    setattr(trace_info.n, jit::Symbol("scale_grad_by_freq"), scale_grad_by_freq);
    setattr(trace_info.n, jit::Symbol("mode"), mode);
    setattr(trace_info.n, jit::Symbol("sparse"), sparse);
  }
  std::tie(result0, result1, result2) = as_variable(baseType->embedding_bag(weight_, indices_, offsets_, scale_grad_by_freq, mode, sparse));
  set_history(result0, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result0, result1, result2 } );
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::embedding_bag_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
  profiler::RecordFunction profiler("embedding_bag_backward");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad, indices, offsets, offset2bag, bag_size )) {
    trace_info = jit::tracer::preRecordTrace( "embedding_bag_backward", { grad, indices, offsets, offset2bag, bag_size } );
    setattr(trace_info.n, jit::Symbol("num_weights"), num_weights);
    setattr(trace_info.n, jit::Symbol("scale_grad_by_freq"), scale_grad_by_freq);
    setattr(trace_info.n, jit::Symbol("mode"), mode);
    setattr(trace_info.n, jit::Symbol("sparse"), sparse);
  }
  auto result = Type::embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, sparse);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::embedding_bag_sparse_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
  profiler::RecordFunction profiler("embedding_bag_sparse_backward");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad, indices, offsets, offset2bag, bag_size )) {
    trace_info = jit::tracer::preRecordTrace( "embedding_bag_sparse_backward", { grad, indices, offsets, offset2bag, bag_size } );
    setattr(trace_info.n, jit::Symbol("num_weights"), num_weights);
    setattr(trace_info.n, jit::Symbol("scale_grad_by_freq"), scale_grad_by_freq);
    setattr(trace_info.n, jit::Symbol("mode"), mode);
  }
  auto result = Type::embedding_bag_sparse_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
  profiler::RecordFunction profiler("embedding_bag_dense_backward");
  auto& grad_ = unpack(grad, "grad", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& offsets_ = unpack(offsets, "offsets", 2);
  auto& offset2bag_ = unpack(offset2bag, "offset2bag", 3);
  auto& bag_size_ = unpack(bag_size, "bag_size", 4);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( grad )) {
    grad_fn = std::make_shared<Error>("the derivative for embedding_bag_dense_backward is not implemented");
    grad_fn->set_next_edges(collect_next_edges( grad ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( grad, indices, offsets, offset2bag, bag_size )) {
    trace_info = jit::tracer::preRecordTrace( "embedding_bag_dense_backward", { grad, indices, offsets, offset2bag, bag_size } );
    setattr(trace_info.n, jit::Symbol("num_weights"), num_weights);
    setattr(trace_info.n, jit::Symbol("scale_grad_by_freq"), scale_grad_by_freq);
    setattr(trace_info.n, jit::Symbol("mode"), mode);
  }
  auto result = as_variable(baseType->embedding_bag_dense_backward(grad_, indices_, offsets_, offset2bag_, bag_size_, num_weights, scale_grad_by_freq, mode));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::expand(const Tensor & self, IntList size) const {
  profiler::RecordFunction profiler("expand");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ExpandBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<ExpandBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "expand", { self } );
    setattr(trace_info.n, jit::Symbol("size"), size);
  }
  auto result = as_view(self, baseType->expand(self_, size));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::expand_as(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("expand_as");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "expand_as", { self, other } );
  
  }
  auto result = Type::expand_as(self, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, bool size_average, bool reduce) const {
  profiler::RecordFunction profiler("hinge_embedding_loss");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, target )) {
    trace_info = jit::tracer::preRecordTrace( "hinge_embedding_loss", { self, target } );
    setattr(trace_info.n, jit::Symbol("margin"), margin);
    setattr(trace_info.n, jit::Symbol("size_average"), size_average);
    setattr(trace_info.n, jit::Symbol("reduce"), reduce);
  }
  auto result = Type::hinge_embedding_loss(self, target, margin, size_average, reduce);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::ger(const Tensor & self, const Tensor & vec2) const {
  profiler::RecordFunction profiler("ger");
  auto& self_ = unpack(self, "self", 0);
  auto& vec2_ = unpack(vec2, "vec2", 1);
  std::shared_ptr<GerBackward> grad_fn;
  if (compute_requires_grad( self, vec2 )) {
    grad_fn = std::make_shared<GerBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, vec2 ));
    grad_fn->vec2_ = SavedVariable(vec2, false);
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, vec2 )) {
    trace_info = jit::tracer::preRecordTrace( "ger", { self, vec2 } );
  
  }
  auto result = as_variable(baseType->ger(self_, vec2_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
  profiler::RecordFunction profiler("ger_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec2_ = unpack(vec2, "vec2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, vec2 )) {
    throw_error_out_requires_grad("ger");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("ger");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, vec2 )) {
    trace_info = jit::tracer::preRecordTrace( "ger_out", { result, self, vec2 } );
  
  }
  baseType->ger_out(result_, self_, vec2_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::index(const Tensor & self, TensorList indices) const {
  profiler::RecordFunction profiler("index");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, indices )) {
    trace_info = jit::tracer::preRecordTrace( "index", flatten( self, indices ) );
  
  }
  auto result = Type::index(self, indices);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::index_put_(Tensor & self, TensorList indices, const Tensor & values) const {
  profiler::RecordFunction profiler("index_put_");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, indices, values )) {
    trace_info = jit::tracer::preRecordTrace( "index_put", flatten( self, indices, values ) );
  
  }
  Type::index_put_(self, indices, values);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
bool VariableType::is_cuda(const Tensor & self) const {
  auto result = Type::is_cuda(self);
  return result;
}
bool VariableType::is_distributed(const Tensor & self) const {
  auto result = Type::is_distributed(self);
  return result;
}
bool VariableType::is_floating_point(const Tensor & self) const {
  profiler::RecordFunction profiler("is_floating_point");
  auto result = Type::is_floating_point(self);
  return result;
}
bool VariableType::is_nonzero(const Tensor & self) const {
  profiler::RecordFunction profiler("is_nonzero");
  auto result = Type::is_nonzero(self);
  return result;
}
bool VariableType::is_same_size(const Tensor & self, const Tensor & other) const {
  auto result = Type::is_same_size(self, other);
  return result;
}
bool VariableType::is_signed(const Tensor & self) const {
  auto result = Type::is_signed(self);
  return result;
}
bool VariableType::is_sparse(const Tensor & self) const {
  auto result = Type::is_sparse(self);
  return result;
}
Tensor VariableType::matmul(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("matmul");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "matmul", { self, other } );
  
  }
  auto result = Type::matmul(self, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::max_pool1d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool1d");
  Tensor result0;
  Tensor result1;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "max_pool1d", { self } );
    setattr(trace_info.n, jit::Symbol("kernel_size"), kernel_size);
    setattr(trace_info.n, jit::Symbol("stride"), stride);
    setattr(trace_info.n, jit::Symbol("padding"), padding);
    setattr(trace_info.n, jit::Symbol("dilation"), dilation);
    setattr(trace_info.n, jit::Symbol("ceil_mode"), ceil_mode);
  }
  std::tie(result0, result1) = Type::max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result0, result1 } );
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::mm(const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("mm");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "mm", { self, mat2 } );
  
  }
  auto result = Type::mm(self, mat2);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("mm_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "mm_out", { result, self, mat2 } );
  
  }
  Type::mm_out(result, self, mat2);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::mv(const Tensor & self, const Tensor & vec) const {
  profiler::RecordFunction profiler("mv");
  auto& self_ = unpack(self, "self", 0);
  auto& vec_ = unpack(vec, "vec", 1);
  std::shared_ptr<MvBackward> grad_fn;
  if (compute_requires_grad( self, vec )) {
    grad_fn = std::make_shared<MvBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, vec ));
    grad_fn->vec_ = SavedVariable(vec, false);
    grad_fn->self_ = SavedVariable(self, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, vec )) {
    trace_info = jit::tracer::preRecordTrace( "mv", { self, vec } );
  
  }
  auto result = as_variable(baseType->mv(self_, vec_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
  profiler::RecordFunction profiler("mv_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec_ = unpack(vec, "vec", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, vec )) {
    throw_error_out_requires_grad("mv");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("mv");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, vec )) {
    trace_info = jit::tracer::preRecordTrace( "mv_out", { result, self, vec } );
  
  }
  baseType->mv_out(result_, self_, vec_);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) const {
  profiler::RecordFunction profiler("narrow");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "narrow", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("start"), start);
    setattr(trace_info.n, jit::Symbol("length"), length);
  }
  auto result = Type::narrow(self, dim, start, length);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::nnpack_spatial_convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t kW, int64_t kH, int64_t padW, int64_t padH) const {
  profiler::RecordFunction profiler("nnpack_spatial_convolution");
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<NnpackSpatialConvolutionBackward> grad_fn;
  if (compute_requires_grad( input, weight, bias )) {
    grad_fn = std::make_shared<NnpackSpatialConvolutionBackward>();
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kW = kW;
    grad_fn->kH = kH;
    grad_fn->padW = padW;
    grad_fn->padH = padH;
    grad_fn->weight_sizes = weight.sizes();
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( input, weight, bias )) {
    trace_info = jit::tracer::preRecordTrace( "nnpack_spatial_convolution", { input, weight, bias } );
    setattr(trace_info.n, jit::Symbol("kW"), kW);
    setattr(trace_info.n, jit::Symbol("kH"), kH);
    setattr(trace_info.n, jit::Symbol("padW"), padW);
    setattr(trace_info.n, jit::Symbol("padH"), padH);
  }
  auto result = as_variable(baseType->nnpack_spatial_convolution(input_, weight_, bias_, kW, kH, padW, padH));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::nnpack_spatial_convolution_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, int64_t kW, int64_t kH, int64_t padW, int64_t padH, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("nnpack_spatial_convolution_backward");
  Tensor result0;
  Tensor result1;
  Tensor result2;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( input, grad_output, weight )) {
    trace_info = jit::tracer::preRecordTrace( "nnpack_spatial_convolution_backward", { input, grad_output, weight } );
    setattr(trace_info.n, jit::Symbol("kW"), kW);
    setattr(trace_info.n, jit::Symbol("kH"), kH);
    setattr(trace_info.n, jit::Symbol("padW"), padW);
    setattr(trace_info.n, jit::Symbol("padH"), padH);
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(result0, result1, result2) = Type::nnpack_spatial_convolution_backward(input, grad_output, weight, kW, kH, padW, padH, output_mask);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result0, result1, result2 } );
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::nnpack_spatial_convolution_backward_input(const Tensor & input, const Tensor & grad_output, const Tensor & weight, int64_t kW, int64_t kH, int64_t padW, int64_t padH) const {
  profiler::RecordFunction profiler("nnpack_spatial_convolution_backward_input");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( input, grad_output, weight )) {
    trace_info = jit::tracer::preRecordTrace( "nnpack_spatial_convolution_backward_input", { input, grad_output, weight } );
    setattr(trace_info.n, jit::Symbol("kW"), kW);
    setattr(trace_info.n, jit::Symbol("kH"), kH);
    setattr(trace_info.n, jit::Symbol("padW"), padW);
    setattr(trace_info.n, jit::Symbol("padH"), padH);
  }
  auto result = Type::nnpack_spatial_convolution_backward_input(input, grad_output, weight, kW, kH, padW, padH);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::nnpack_spatial_convolution_backward_weight(const Tensor & input, IntList weight_size, const Tensor & grad_output, int64_t kW, int64_t kH, int64_t padW, int64_t padH) const {
  profiler::RecordFunction profiler("nnpack_spatial_convolution_backward_weight");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( input, grad_output )) {
    trace_info = jit::tracer::preRecordTrace( "nnpack_spatial_convolution_backward_weight", { input, grad_output } );
    setattr(trace_info.n, jit::Symbol("weight_size"), weight_size);
    setattr(trace_info.n, jit::Symbol("kW"), kW);
    setattr(trace_info.n, jit::Symbol("kH"), kH);
    setattr(trace_info.n, jit::Symbol("padW"), padW);
    setattr(trace_info.n, jit::Symbol("padH"), padH);
  }
  auto result = Type::nnpack_spatial_convolution_backward_weight(input, weight_size, grad_output, kW, kH, padW, padH);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::ones_like(const Tensor & self) const {
  profiler::RecordFunction profiler("ones_like");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "ones_like", { self } );
  
  }
  auto result = Type::ones_like(self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::ones_like(const Tensor & self, const Type & dtype) const {
  profiler::RecordFunction profiler("ones_like");
  auto result = Type::ones_like(self, dtype);
  return result;
}
Tensor VariableType::permute(const Tensor & self, IntList dims) const {
  profiler::RecordFunction profiler("permute");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PermuteBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<PermuteBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dims = dims;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "permute", { self } );
    setattr(trace_info.n, jit::Symbol("dims"), dims);
  }
  auto result = as_view(self, baseType->permute(self_, dims));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::pin_memory(const Tensor & self) const {
  profiler::RecordFunction profiler("pin_memory");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "pin_memory", { self } );
  
  }
  auto result = Type::pin_memory(self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::rand_like(const Tensor & self) const {
  profiler::RecordFunction profiler("rand_like");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "rand_like", { self } );
  
  }
  auto result = Type::rand_like(self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::rand_like(const Tensor & self, const Type & dtype) const {
  profiler::RecordFunction profiler("rand_like");
  auto result = Type::rand_like(self, dtype);
  return result;
}
Tensor VariableType::randn_like(const Tensor & self) const {
  profiler::RecordFunction profiler("randn_like");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "randn_like", { self } );
  
  }
  auto result = Type::randn_like(self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::randn_like(const Tensor & self, const Type & dtype) const {
  profiler::RecordFunction profiler("randn_like");
  auto result = Type::randn_like(self, dtype);
  return result;
}
Tensor VariableType::repeat(const Tensor & self, IntList repeats) const {
  profiler::RecordFunction profiler("repeat");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<RepeatBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<RepeatBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->repeats = repeats;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "repeat", { self } );
    setattr(trace_info.n, jit::Symbol("repeats"), repeats);
  }
  auto result = as_variable(baseType->repeat(self_, repeats));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
  profiler::RecordFunction profiler("RoiPooling2d_forward");
  auto& input_ = unpack(input, "input", 0);
  auto& rois_ = unpack(rois, "rois", 1);
  check_no_requires_grad(rois, "rois");
  std::shared_ptr<Roipooling2DBackward> grad_fn;
  if (compute_requires_grad( input )) {
    grad_fn = std::make_shared<Roipooling2DBackward>();
    grad_fn->set_next_edges(collect_next_edges( input ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->rois_ = SavedVariable(rois, false);
    grad_fn->pooledHeight = pooledHeight;
    grad_fn->pooledWidth = pooledWidth;
    grad_fn->spatialScale = spatialScale;
  }
  Tensor result0;
  Tensor result1;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( input, rois )) {
    trace_info = jit::tracer::preRecordTrace( "RoiPooling2d_forward", { input, rois } );
    setattr(trace_info.n, jit::Symbol("pooledHeight"), pooledHeight);
    setattr(trace_info.n, jit::Symbol("pooledWidth"), pooledWidth);
    setattr(trace_info.n, jit::Symbol("spatialScale"), spatialScale);
  }
  std::tie(result0, result1) = as_variable(baseType->RoiPooling2d_forward(input_, rois_, pooledHeight, pooledWidth, spatialScale));
  set_history(result0, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result0, result1 } );
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
  profiler::RecordFunction profiler("RoiPooling2d_backward");
  auto& input_ = unpack(input, "input", 0);
  auto& rois_ = unpack(rois, "rois", 1);
  auto& gradOutput_ = unpack(gradOutput, "gradOutput", 5);
  auto& argmaxes_ = unpack(argmaxes, "argmaxes", 6);
  std::shared_ptr<Error> grad_fn;
  if (compute_requires_grad( input, rois, gradOutput, argmaxes )) {
    grad_fn = std::make_shared<Error>("the derivative for RoiPooling2d_backward is not implemented");
    grad_fn->set_next_edges(collect_next_edges( input, rois, gradOutput, argmaxes ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( input, rois, gradOutput, argmaxes )) {
    trace_info = jit::tracer::preRecordTrace( "RoiPooling2d_backward", { input, rois, gradOutput, argmaxes } );
    setattr(trace_info.n, jit::Symbol("pooledHeight"), pooledHeight);
    setattr(trace_info.n, jit::Symbol("pooledWidth"), pooledWidth);
    setattr(trace_info.n, jit::Symbol("spatialScale"), spatialScale);
  }
  auto result = as_variable(baseType->RoiPooling2d_backward(input_, rois_, pooledHeight, pooledWidth, spatialScale, gradOutput_, argmaxes_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("rrelu");
  auto result = Type::rrelu(self, lower, upper, training, generator);
  return result;
}
Tensor & VariableType::rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("rrelu_");
  Type::rrelu_(self, lower, upper, training, generator);
  return self;
}
Tensor VariableType::select(const Tensor & self, int64_t dim, int64_t index) const {
  profiler::RecordFunction profiler("select");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "select", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("index"), index);
  }
  auto result = Type::select(self, dim, index);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::selu(const Tensor & self) const {
  profiler::RecordFunction profiler("selu");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "selu", { self } );
  
  }
  auto result = Type::selu(self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::selu_(Tensor & self) const {
  profiler::RecordFunction profiler("selu_");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "selu", { self } );
  
  }
  Type::selu_(self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
int64_t VariableType::size(const Tensor & self, int64_t dim) const {
  auto result = Type::size(self, dim);
  return result;
}
Tensor VariableType::slice(const Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step) const {
  profiler::RecordFunction profiler("slice");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "slice", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
    setattr(trace_info.n, jit::Symbol("start"), start);
    setattr(trace_info.n, jit::Symbol("end"), end);
    setattr(trace_info.n, jit::Symbol("step"), step);
  }
  auto result = Type::slice(self, dim, start, end, step);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::smm(const Tensor & self, const Tensor & mat2) const {
  profiler::RecordFunction profiler("smm");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "smm", { self, mat2 } );
  
  }
  auto result = Type::smm(self, mat2);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
std::vector<Tensor> VariableType::split(const Tensor & self, int64_t split_size, int64_t dim) const {
  profiler::RecordFunction profiler("split");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SplitBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SplitBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->split_size = split_size;
    grad_fn->dim = dim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "split", { self } );
    setattr(trace_info.n, jit::Symbol("split_size"), split_size);
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto result = as_variable(baseType->split(self_, split_size, dim));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  flatten(result) );
  }
  return result;
}
Tensor VariableType::squeeze(const Tensor & self) const {
  profiler::RecordFunction profiler("squeeze");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SqueezeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SqueezeBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "squeeze", { self } );
  
  }
  auto result = as_view(self, baseType->squeeze(self_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::squeeze(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("squeeze");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SqueezeBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SqueezeBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dim = dim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "squeeze", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto result = as_view(self, baseType->squeeze(self_, dim));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::squeeze_(Tensor & self) const {
  profiler::RecordFunction profiler("squeeze_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SqueezeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SqueezeBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "squeeze", { self } );
  
  }
  baseType->squeeze_(self_);
  increment_version(self);
  set_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::squeeze_(Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("squeeze_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SqueezeBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<SqueezeBackward1>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
    grad_fn->dim = dim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "squeeze", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->squeeze_(self_, dim);
  increment_version(self);
  set_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor VariableType::sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("sspaddmm");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, mat1, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "sspaddmm", { self, mat1, mat2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  auto result = Type::sspaddmm(self, mat1, mat2, beta, alpha);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("sspaddmm_out");
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat1_ = unpack(mat1, "mat1", 2);
  auto& mat2_ = unpack(mat2, "mat2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    throw_error_out_requires_grad("sspaddmm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sspaddmm");
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, self, mat1, mat2 )) {
    trace_info = jit::tracer::preRecordTrace( "sspaddmm_out", { result, self, mat1, mat2 } );
    setattr(trace_info.n, jit::Symbol("beta"), beta);
    setattr(trace_info.n, jit::Symbol("alpha"), alpha);
  }
  baseType->sspaddmm_out(result_, self_, mat1_, mat2_, beta, alpha);
  increment_version(result);
  rebase_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::stack(TensorList tensors, int64_t dim) const {
  profiler::RecordFunction profiler("stack");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( tensors )) {
    trace_info = jit::tracer::preRecordTrace( "stack", flatten( tensors ) );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto result = Type::stack(tensors, dim);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::stack_out(Tensor & result, TensorList tensors, int64_t dim) const {
  profiler::RecordFunction profiler("stack_out");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( result, tensors )) {
    trace_info = jit::tracer::preRecordTrace( "stack_out", flatten( result, tensors ) );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  Type::stack_out(result, tensors, dim);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  {result} );
  }
  return result;
}
Tensor VariableType::stft(const Tensor & self, int64_t frame_length, int64_t hop, int64_t fft_size, bool return_onesided, const Tensor & window, int64_t pad_end) const {
  profiler::RecordFunction profiler("stft");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, window )) {
    trace_info = jit::tracer::preRecordTrace( "stft", { self, window } );
    setattr(trace_info.n, jit::Symbol("frame_length"), frame_length);
    setattr(trace_info.n, jit::Symbol("hop"), hop);
    setattr(trace_info.n, jit::Symbol("fft_size"), fft_size);
    setattr(trace_info.n, jit::Symbol("return_onesided"), return_onesided);
    setattr(trace_info.n, jit::Symbol("pad_end"), pad_end);
  }
  auto result = Type::stft(self, frame_length, hop, fft_size, return_onesided, window, pad_end);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
int64_t VariableType::stride(const Tensor & self, int64_t dim) const {
  auto result = Type::stride(self, dim);
  return result;
}
Tensor & VariableType::transpose_(Tensor & self, int64_t dim0, int64_t dim1) const {
  profiler::RecordFunction profiler("transpose_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TransposeBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TransposeBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim0 = dim0;
    grad_fn->dim1 = dim1;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "transpose", { self } );
    setattr(trace_info.n, jit::Symbol("dim0"), dim0);
    setattr(trace_info.n, jit::Symbol("dim1"), dim1);
  }
  baseType->transpose_(self_, dim0, dim1);
  increment_version(self);
  set_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor & VariableType::t_(Tensor & self) const {
  profiler::RecordFunction profiler("t_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<TBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "t", { self } );
  
  }
  baseType->t_(self_);
  increment_version(self);
  set_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor VariableType::type_as(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("type_as");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "type_as", { self, other } );
  
  }
  auto result = Type::type_as(self, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::_unsafe_view(const Tensor & self, IntList size) const {
  profiler::RecordFunction profiler("_unsafe_view");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UnsafeViewBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<UnsafeViewBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes();
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "_unsafe_view", { self } );
    setattr(trace_info.n, jit::Symbol("size"), size);
  }
  auto result = as_variable(baseType->_unsafe_view(self_, size));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::unsqueeze(const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("unsqueeze");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UnsqueezeBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<UnsqueezeBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "unsqueeze", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  auto result = as_view(self, baseType->unsqueeze(self_, dim));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor & VariableType::unsqueeze_(Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("unsqueeze_");
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<UnsqueezeBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<UnsqueezeBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "unsqueeze", { self } );
    setattr(trace_info.n, jit::Symbol("dim"), dim);
  }
  baseType->unsqueeze_(self_, dim);
  increment_version(self);
  set_history(self, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { self } );
  }
  return self;
}
Tensor VariableType::view_as(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("view_as");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, other )) {
    trace_info = jit::tracer::preRecordTrace( "view_as", { self, other } );
  
  }
  auto result = Type::view_as(self, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("where");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( condition, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "where", { condition, self, other } );
  
  }
  auto result = Type::where(condition, self, other);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_s_where");
  auto& condition_ = unpack(condition, "condition", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<SWhereBackward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::make_shared<SWhereBackward>();
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->condition_ = SavedVariable(condition, false);
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( condition, self, other )) {
    trace_info = jit::tracer::preRecordTrace( "_s_where", { condition, self, other } );
  
  }
  auto result = as_variable(baseType->_s_where(condition_, self_, other_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::zeros_like(const Tensor & self) const {
  profiler::RecordFunction profiler("zeros_like");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self )) {
    trace_info = jit::tracer::preRecordTrace( "zeros_like", { self } );
  
  }
  auto result = Type::zeros_like(self);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::zeros_like(const Tensor & self, const Type & dtype) const {
  profiler::RecordFunction profiler("zeros_like");
  auto result = Type::zeros_like(self, dtype);
  return result;
}
Tensor VariableType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
  profiler::RecordFunction profiler("_standard_gamma_grad");
  auto& self_ = unpack(self, "self", 0);
  auto& output_ = unpack(output, "output", 1);
  std::shared_ptr<StandardGammaGradBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<StandardGammaGradBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( self, output )) {
    trace_info = jit::tracer::preRecordTrace( "_standard_gamma_grad", { self, output } );
  
  }
  auto result = as_variable(baseType->_standard_gamma_grad(self_, output_));
  set_history(result, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
Tensor VariableType::poisson(const Tensor & self, Generator * generator) const {
  profiler::RecordFunction profiler("poisson");
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PoissonBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::make_shared<PoissonBackward>();
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  auto result = as_variable(baseType->poisson(self_, generator));
  set_history(result, grad_fn);
  return result;
}
Tensor VariableType::_cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional) const {
  profiler::RecordFunction profiler("_cudnn_rnn_flatten_weight");
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( weight_arr )) {
    trace_info = jit::tracer::preRecordTrace( "_cudnn_rnn_flatten_weight", flatten( weight_arr ) );
    setattr(trace_info.n, jit::Symbol("weight_stride0"), weight_stride0);
    setattr(trace_info.n, jit::Symbol("input_size"), input_size);
    setattr(trace_info.n, jit::Symbol("mode"), mode);
    setattr(trace_info.n, jit::Symbol("hidden_size"), hidden_size);
    setattr(trace_info.n, jit::Symbol("num_layers"), num_layers);
    setattr(trace_info.n, jit::Symbol("batch_first"), batch_first);
    setattr(trace_info.n, jit::Symbol("bidirectional"), bidirectional);
  }
  auto result = Type::_cudnn_rnn_flatten_weight(weight_arr, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result } );
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> VariableType::_cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, const Tensor & dropout_state) const {
  profiler::RecordFunction profiler("_cudnn_rnn");
  auto& input_ = unpack(input, "input", 0);
  auto weight_ = unpack(weight, "weight", 1);
  auto weight_buf_ = unpack_opt(weight_buf, "weight_buf", 3);
  auto& hx_ = unpack(hx, "hx", 4);
  auto cx_ = unpack_opt(cx, "cx", 5);
  auto dropout_state_ = unpack_opt(dropout_state, "dropout_state", 14);
  check_no_requires_grad(weight_buf, "weight_buf");
  std::shared_ptr<CudnnRnnBackward> grad_fn;
  if (compute_requires_grad( input, weight, hx, cx )) {
    grad_fn = std::make_shared<CudnnRnnBackward>();
    grad_fn->set_next_edges(collect_next_edges( input, weight, hx, cx ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = make_saved_variable_list(weight);
    grad_fn->weight_stride0 = weight_stride0;
    grad_fn->hx_ = SavedVariable(hx, false);
    grad_fn->cx_ = SavedVariable(cx, false);
    grad_fn->mode = mode;
    grad_fn->hidden_size = hidden_size;
    grad_fn->num_layers = num_layers;
    grad_fn->batch_first = batch_first;
    grad_fn->dropout = dropout;
    grad_fn->train = train;
    grad_fn->bidirectional = bidirectional;
    grad_fn->batch_sizes = batch_sizes;
    grad_fn->dropout_state_ = SavedVariable(dropout_state, false);
    grad_fn->weight_size_ = weight.size();
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  Tensor result4;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( input, weight, weight_buf, hx, cx, dropout_state )) {
    trace_info = jit::tracer::preRecordTrace( "_cudnn_rnn", flatten( input, weight, weight_buf, hx, cx, dropout_state ) );
    setattr(trace_info.n, jit::Symbol("weight_stride0"), weight_stride0);
    setattr(trace_info.n, jit::Symbol("mode"), mode);
    setattr(trace_info.n, jit::Symbol("hidden_size"), hidden_size);
    setattr(trace_info.n, jit::Symbol("num_layers"), num_layers);
    setattr(trace_info.n, jit::Symbol("batch_first"), batch_first);
    setattr(trace_info.n, jit::Symbol("dropout"), dropout);
    setattr(trace_info.n, jit::Symbol("train"), train);
    setattr(trace_info.n, jit::Symbol("bidirectional"), bidirectional);
    setattr(trace_info.n, jit::Symbol("batch_sizes"), batch_sizes);
  }
  std::tie(result0, result1, result2, result3, result4) = as_variable(baseType->_cudnn_rnn(input_, weight_, weight_stride0, weight_buf_, hx_, cx_, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state_));
  set_history({ result0, result1, result2 }, grad_fn);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  { result0, result1, result2, result3, result4 } );
  }
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result3_ = SavedVariable(result3, true);
    grad_fn->result4_ = SavedVariable(result4, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> VariableType::_cudnn_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask) const {
  profiler::RecordFunction profiler("_cudnn_rnn_backward");
  Tensor result0;
  Tensor result1;
  Tensor result2;
  std::vector<Tensor> result3;
  jit::tracer::PreTraceInfo trace_info;
  if (jit::tracer::isTracing( input, weight, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, dropout_state, reserve )) {
    trace_info = jit::tracer::preRecordTrace( "_cudnn_rnn_backward", flatten( input, weight, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, dropout_state, reserve ) );
    setattr(trace_info.n, jit::Symbol("weight_stride0"), weight_stride0);
    setattr(trace_info.n, jit::Symbol("mode"), mode);
    setattr(trace_info.n, jit::Symbol("hidden_size"), hidden_size);
    setattr(trace_info.n, jit::Symbol("num_layers"), num_layers);
    setattr(trace_info.n, jit::Symbol("batch_first"), batch_first);
    setattr(trace_info.n, jit::Symbol("dropout"), dropout);
    setattr(trace_info.n, jit::Symbol("train"), train);
    setattr(trace_info.n, jit::Symbol("bidirectional"), bidirectional);
    setattr(trace_info.n, jit::Symbol("batch_sizes"), batch_sizes);
    setattr(trace_info.n, jit::Symbol("output_mask"), output_mask);
  }
  std::tie(result0, result1, result2, result3) = Type::_cudnn_rnn_backward(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
  if (trace_info.state != nullptr) {
    jit::tracer::postRecordTrace( trace_info,  flatten( result0, result1, result2, result3 ) );
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}

}} // namespace torch::autograd
