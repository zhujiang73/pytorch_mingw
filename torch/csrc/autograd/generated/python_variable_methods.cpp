// generated from tools/autograd/templates/python_variable_methods.cpp

#include <Python.h>

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Size.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/cuda/lazy_init.h"
#ifdef WITH_CUDA
#include "torch/csrc/cuda/Stream.h"
#endif
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/python_tuples.h"
#include "torch/csrc/utils/tensor_apply.h"
#include "torch/csrc/utils/tensor_list.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/utils/tensor_types.h"

#include "python_variable_methods_dispatch.h"

using at::Tensor;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static PyObject * THPVariable_apply_(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.requires_grad()) {
    throw std::runtime_error(
        "Can't call apply_() on Variable that requires grad. Use "
        "var.detach().apply_() instead.");
  }
  return THPVariable_Wrap(torch::utils::apply_(self_, arg));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_clamp(const Tensor & self, Scalar min, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp(min, max);
}
static Tensor dispatch_clamp_min(const Tensor & self, Scalar min) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_min(min);
}
static Tensor dispatch_clamp_max(const Tensor & self, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_max(max);
}

static PyObject * THPVariable_clamp(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp(Scalar min=None, Scalar max=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (!r.isNone(0) && !r.isNone(1)) {
    return THPVariable_Wrap(dispatch_clamp(self_, r.scalar(0), r.scalar(1)));
  } else if (!r.isNone(0)) {
    return THPVariable_Wrap(dispatch_clamp_min(self_, r.scalar(0)));
  } else if (!r.isNone(1)) {
    return THPVariable_Wrap(dispatch_clamp_max(self_, r.scalar(1)));
  } else {
    throw std::runtime_error("At least one of 'min' or 'max' must not be None");
  }
  END_HANDLE_TH_ERRORS
}

static Tensor & dispatch_clamp_(Tensor & self, Scalar min, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_(min, max);
}
static Tensor & dispatch_clamp_min_(Tensor & self, Scalar min) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_min_(min);
}
static Tensor & dispatch_clamp_max_(Tensor & self, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_max_(max);
}

static PyObject * THPVariable_clamp_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_(Scalar min=None, Scalar max=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (!r.isNone(0) && !r.isNone(1)) {
    return THPVariable_Wrap(dispatch_clamp_(self_, r.scalar(0), r.scalar(1)));
  } else if (!r.isNone(0)) {
    return THPVariable_Wrap(dispatch_clamp_min_(self_, r.scalar(0)));
  } else if (!r.isNone(1)) {
    return THPVariable_Wrap(dispatch_clamp_max_(self_, r.scalar(1)));
  } else {
    throw std::runtime_error("At least one of 'min' or 'max' must not be None");
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_size(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "size(int64_t dim)",
    "size()",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(self_.size(r.toInt64(0)));
  } else if (r.idx == 1) {
    // Yes, this is called sizes in ATen
    IntList sizes = self_.sizes();
    // we can't do the normal wrapping here because IntList maps to both
    // torch.Size and tuple in python.
    return THPSize_New(sizes.size(), sizes.data());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_stride(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stride(int64_t dim)",
    "stride()",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(self_.stride(r.toInt64(0)));
  } else if (r.idx == 1) {
    // yes, this is called strides in ATen.
    IntList strides = self_.strides();
    // we can't do the normal wrapping here because IntList maps to both
    // torch.Size and tuple in python
    return THPUtils_packInt64Array(strides.size(), strides.data());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_dim(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
   return THPUtils_packInt64(self_.dim());
   END_HANDLE_TH_ERRORS
}

static Tensor dispatch_contiguous(const Tensor & self) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.contiguous();
}

static PyObject * THPVariable_contiguous(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  // avoids touching the GIL or current device if self is already contiguous
  if (self_.is_contiguous()) {
    Py_INCREF(self);
    return self;
  }
  return THPVariable_Wrap(dispatch_contiguous(self_));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_copy_(Tensor & self, const Tensor & other, bool non_blocking) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.copy_(other, non_blocking);
}

static PyObject * THPVariable_copy_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "copy_(Tensor other, bool non_blocking=False)",
    "copy_(Tensor other, bool async=False)|deprecated"
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  return THPVariable_Wrap(dispatch_copy_(self_, r.tensor(0), r.toBool(1)));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_detach(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return THPVariable_Wrap(self_.detach());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_detach_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  reinterpret_cast<THPVariable*>(self)->cdata.detach_();
  Py_INCREF(self);
  return self;
  END_HANDLE_TH_ERRORS
}

static double dispatch_to_CDouble(const Tensor & self) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.toCDouble();
}

static int64_t dispatch_to_CLong(const Tensor & self) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.toCLong();
}

static PyObject * THPVariable_float_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_to_CDouble(self_));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_integral_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (isFloatingType(self_.type().scalarType())) {
    // we can't dispatch to toCLong here because we want to avoid ATen overflow checks;
    // the python integral type (long in python2) can't overflow.
    return THPUtils_packDoubleAsInt(dispatch_to_CDouble(self_));
  } else {
    return wrap(dispatch_to_CLong(self_));
  }
  END_HANDLE_TH_ERRORS
}

// This is the __index__ function in Python which is similar to __int__, but
// called when used as a slice.
static PyObject * THPVariable_index_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  // TODO: change the condition to `self_.dim() != 0` once we expose scalars
  // in PyTorch.
  if (!isIntegralType(self_.type().scalarType()) || self_.numel() != 1) {
    throw TypeError("only integer tensors of a single element can be converted to an index");
  }
  return wrap(dispatch_to_CLong(self_));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_invert(const Tensor & self) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return 1 - self;
}

static PyObject * THPVariable_invert(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.type().scalarType() != at::kByte) {
    throw TypeError("~ (operator.invert) is only implemented on byte tensors");
  }
  return THPVariable_Wrap(dispatch_invert(self_));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_type(const Tensor & self, const at::Type & type, int device, bool non_blocking) {
#ifdef WITH_CUDA
  if (type.is_cuda()) {
    torch::cuda::lazy_init();
  }
#endif
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  int64_t tensor_device = self.is_cuda() ? self.get_device() : -1;
  if (self.is_cuda() && type.is_cuda() && tensor_device != at::current_device()) {
    // copy if the devices are different even if the types are the same
    return type.copy(self, non_blocking);
  }
  return self.toType(type, non_blocking);
}

static Tensor dispatch_type(const Tensor & self, const at::Type & type) {
  int64_t device = self.is_cuda() ? self.get_device() : -1;
  return dispatch_type(self, type, device, false);
}

static PyObject * THPVariable_cpu(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
   auto backend = self_.is_sparse() ? Backend::SparseCPU : Backend::CPU;
   auto& type = self_.type().toBackend(backend);
   return wrap(dispatch_type(self_, type));
   END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_cuda(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cuda(int64_t? device=-1, bool non_blocking=False)",
    "cuda(int64_t? device=-1, bool async=False)|deprecated"
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  auto backend = self_.is_sparse() ? at::kSparseCUDA : at::kCUDA;
  auto& type = self_.type().toBackend(backend);
  auto device = r.toInt64(0);
  return THPVariable_Wrap(dispatch_type(self_, type, device, r.toBool(1)));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_to_type(PyObject* self, ScalarType scalarType) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  auto& type = self_.type().toScalarType(scalarType);
  return THPVariable_Wrap(dispatch_type(self_, type));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_byte(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Byte);
}

static PyObject * THPVariable_char(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Char);
}

static PyObject * THPVariable_double(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Double);
}

static PyObject * THPVariable_float(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Float);
}

static PyObject * THPVariable_half(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Half);
}

static PyObject * THPVariable_int(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Int);
}

static PyObject * THPVariable_long(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Long);
}

static PyObject * THPVariable_short(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Short);
}

static PyObject * THPVariable_element_size(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  size_t element_size = self_.type().elementSizeInBytes();
  return THPUtils_packInt64(element_size);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_numpy(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.requires_grad()) {
    throw std::runtime_error(
        "Can't call numpy() on Variable that requires grad. "
        "Use var.detach().numpy() instead.");
  }
  return torch::utils::tensor_to_numpy(self_.data());
  END_HANDLE_TH_ERRORS
}

// TODO: move this to ATen. We would need to expose Stream objects in ATen.
static PyObject * THPVariable_record_stream(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
#ifdef WITH_CUDA
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (!THCPStream_Check(arg)) {
    return PyErr_Format(PyExc_TypeError, "expected Stream object");
  }
  void* data = self_.data_ptr();
  THCCachingAllocator_recordStream(data, ((THCPStream*)arg)->cdata);
  Py_RETURN_NONE;
#else
  throw std::runtime_error("PyTorch compiled without CUDA support");
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_item(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.is_floating_point()) {
    return wrap(dispatch_to_CDouble(self_));
  } else {
    return wrap(dispatch_to_CLong(self_));
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_map_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({ "map_(Tensor other, PyObject* callable)" });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  Variable other = r.tensor(0);
  if (self_.requires_grad() || other.requires_grad()) {
    throw std::runtime_error(
        "Can't call map_() on Variable that requires grad. Use "
        "var.detach().map_() instead.");
  }
  return THPVariable_Wrap(torch::utils::map_(self_, other, r.pyobject(1)));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_map2_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({ "map2_(Tensor x, Tensor y, PyObject* callable)" });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  Variable x = r.tensor(0);
  Variable y = r.tensor(1);
  if (self_.requires_grad() || x.requires_grad() || y.requires_grad()) {
    throw std::runtime_error(
        "Can't call map2_() on Variable that requires grad. Use "
        "var.detach().map2_() instead.");
  }
  return THPVariable_Wrap(torch::utils::map2_(self_, x, y, r.pyobject(2)));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  AutoGPU auto_gpu(self_);
  return THPVariable_Wrap(torch::utils::legacy_tensor_ctor(self_.type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  AutoGPU auto_gpu(self_);
  return THPVariable_Wrap(torch::utils::new_tensor(self_.type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_storage(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return createPyObject(*self_.storage());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_storage_type(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  auto storage = THPObjectPtr(createPyObject(*self_.storage()));
  auto storage_type = (PyObject*)Py_TYPE(storage);
  Py_INCREF(storage_type);
  return storage_type;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_tolist(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return torch::utils::tensor_to_list(self_.data());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_type(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "type(PyObject* new_type=None, bool non_blocking=False)",
    "type(PyObject* new_type=None, bool async=False)|deprecated"
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.isNone(0)) {
    return THPUtils_packString(torch::utils::type_to_string(self_.type()));
  }
  auto obj = r.pyobject(0);
  std::string type_name;
  if (PyType_Check(obj)) {
    type_name = ((PyTypeObject*)obj)->tp_name;
  } else if (THPUtils_checkString(obj)) {
    type_name = THPUtils_unpackString(obj);
  } else {
    throw TypeError("new_type must be a type or str object");
  }
  auto& type = torch::utils::type_from_string(type_name);
  return THPVariable_Wrap(dispatch_type(self_, type, -1, r.toBool(1)));
  END_HANDLE_TH_ERRORS
}

// generated methods start here

static PyObject * THPVariable___and__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__and__(Scalar other)",
    "__and__(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___and__(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___and__(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___iand__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__iand__(Scalar other)",
    "__iand__(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___iand__(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___iand__(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___ilshift__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__ilshift__(Scalar other)",
    "__ilshift__(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___ilshift__(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___ilshift__(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___ior__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__ior__(Scalar other)",
    "__ior__(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___ior__(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___ior__(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___irshift__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__irshift__(Scalar other)",
    "__irshift__(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___irshift__(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___irshift__(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___ixor__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__ixor__(Scalar other)",
    "__ixor__(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___ixor__(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___ixor__(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___lshift__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__lshift__(Scalar other)",
    "__lshift__(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___lshift__(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___lshift__(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___or__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__or__(Scalar other)",
    "__or__(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___or__(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___or__(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___rshift__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__rshift__(Scalar other)",
    "__rshift__(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___rshift__(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___rshift__(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___xor__(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__xor__(Scalar other)",
    "__xor__(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch___xor__(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___xor__(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__addmv(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_addmv(Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__addmv(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__addmv_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_addmv_(Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__addmv_(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__addr(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_addr(Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__addr(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__addr_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_addr_(Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__addr_(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__copy_ignoring_overlaps_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_copy_ignoring_overlaps_(Tensor src)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__copy_ignoring_overlaps_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__det_with_svd(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch__det_with_svd(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__dimI(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch__dimI(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__dimV(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch__dimV(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__dot(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_dot(Tensor tensor)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__dot(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__ger(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_ger(Tensor vec2)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__ger(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__indices(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch__indices(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__mm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mm(Tensor mat2)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__mm(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__mv(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mv(Tensor vec)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__mv(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__nnz(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch__nnz(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__s_where(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_s_where(Tensor condition, Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__s_where(r.tensor(0), self_, r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__sparse_mask(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_mask(Tensor mask)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__sparse_mask(self_, SparseTensor(r.tensor(0))));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__standard_gamma(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_standard_gamma(*, Generator generator=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__standard_gamma(self_, r.generator(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__standard_gamma_grad(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_standard_gamma_grad(Tensor output)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch__standard_gamma_grad(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__values(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch__values(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_abs(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_abs(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_abs_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_abs_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_acos(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_acos(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_acos_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_acos_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_add(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "add(Scalar alpha, Tensor other)|deprecated",
    "add(Scalar other, *, Scalar alpha=1)",
    "add(Tensor other, *, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_add(self_, r.scalar(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_add(self_, r.scalar(0), r.scalar(1)));
  } else if (r.idx == 2) {
    return wrap(dispatch_add(self_, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_add_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "add_(Scalar alpha, Tensor other)|deprecated",
    "add_(Scalar other, *, Scalar alpha=1)",
    "add_(Tensor other, *, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_add_(self_, r.scalar(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_add_(self_, r.scalar(0), r.scalar(1)));
  } else if (r.idx == 2) {
    return wrap(dispatch_add_(self_, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addbmm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addbmm(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addbmm(r.scalar(0), self_, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addbmm(r.scalar(0), self_, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addbmm(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addbmm_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addbmm_(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm_(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm_(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addbmm_(r.scalar(0), self_, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addbmm_(r.scalar(0), self_, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addbmm_(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addcdiv(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addcdiv(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcdiv(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addcdiv(self_, r.scalar(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addcdiv(self_, r.tensor(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addcdiv_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addcdiv_(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcdiv_(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addcdiv_(self_, r.scalar(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addcdiv_(self_, r.tensor(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addcmul(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addcmul(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcmul(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addcmul(self_, r.scalar(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addcmul(self_, r.tensor(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addcmul_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addcmul_(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcmul_(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addcmul_(self_, r.scalar(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addcmul_(self_, r.tensor(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmm(Scalar beta, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "addmm(Scalar beta, Tensor mat1, Tensor mat2)|deprecated",
    "addmm(Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addmm(r.scalar(0), self_, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addmm(r.scalar(0), self_, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addmm(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmm_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmm_(Scalar beta, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "addmm_(Scalar beta, Tensor mat1, Tensor mat2)|deprecated",
    "addmm_(Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addmm_(r.scalar(0), self_, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addmm_(r.scalar(0), self_, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addmm_(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmv(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmv(Scalar beta, Scalar alpha, Tensor mat, Tensor vec)|deprecated",
    "addmv(Scalar beta, Tensor mat, Tensor vec)|deprecated",
    "addmv(Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addmv(r.scalar(0), self_, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addmv(r.scalar(0), self_, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addmv(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmv_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmv_(Scalar beta, Scalar alpha, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Scalar beta, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addmv_(r.scalar(0), self_, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addmv_(r.scalar(0), self_, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addmv_(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addr(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addr(Scalar beta, Scalar alpha, Tensor vec1, Tensor vec2)|deprecated",
    "addr(Scalar beta, Tensor vec1, Tensor vec2)|deprecated",
    "addr(Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addr(r.scalar(0), self_, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addr(r.scalar(0), self_, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addr(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addr_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addr_(Scalar beta, Scalar alpha, Tensor vec1, Tensor vec2)|deprecated",
    "addr_(Scalar beta, Tensor vec1, Tensor vec2)|deprecated",
    "addr_(Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_addr_(r.scalar(0), self_, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addr_(r.scalar(0), self_, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addr_(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_all(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_all(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_allclose(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "allclose(Tensor other, double rtol=1e-05, double atol=1e-08)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_allclose(self_, r.tensor(0), r.toDouble(1), r.toDouble(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_any(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_any(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_as_strided(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "as_strided(IntList size, IntList stride, int64_t storage_offset=-1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_as_strided(self_, r.intlist(0), r.intlist(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_as_strided_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "as_strided_(IntList size, IntList stride, int64_t storage_offset=-1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_as_strided_(self_, r.intlist(0), r.intlist(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_asin(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_asin(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_asin_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_asin_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_atan(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_atan(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_atan2(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan2(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_atan2(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_atan2_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan2_(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_atan2_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_atan_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_atan_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_baddbmm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "baddbmm(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_baddbmm(r.scalar(0), self_, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_baddbmm(r.scalar(0), self_, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_baddbmm(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_baddbmm_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "baddbmm_(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm_(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm_(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_baddbmm_(r.scalar(0), self_, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_baddbmm_(r.scalar(0), self_, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_baddbmm_(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bernoulli(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bernoulli(*, Generator generator=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_bernoulli(self_, r.generator(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bernoulli_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bernoulli_(Tensor p, Generator generator=None)",
    "bernoulli_(double p=0.5, Generator generator=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_bernoulli_(self_, r.tensor(0), r.generator(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_bernoulli_(self_, r.toDouble(0), r.generator(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bmm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bmm(Tensor mat2)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_bmm(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_btrifact(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "btrifact(*, bool pivot=True)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_btrifact(self_, r.toBool(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_btrifact_with_info(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "btrifact_with_info(*, bool pivot=True)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_btrifact_with_info(self_, r.toBool(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_btrisolve(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "btrisolve(Tensor LU_data, Tensor LU_pivots)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_btrisolve(self_, r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cauchy_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cauchy_(double median=0, double sigma=1, *, Generator generator=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cauchy_(self_, r.toDouble(0), r.toDouble(1), r.generator(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ceil(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_ceil(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ceil_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_ceil_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_chunk(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "chunk(int64_t chunks, int64_t dim=0)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_chunk(self_, r.toInt64(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clone(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_clone(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_coalesce(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_coalesce(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_conv_tbc(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_tbc(Tensor weight, Tensor bias, int64_t pad)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_conv_tbc(self_, r.tensor(0), r.tensor(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cos(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_cos(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cos_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_cos_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cosh(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_cosh(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cosh_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_cosh_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cross(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cross(Tensor other, int64_t dim=-1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cross(self_, r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cumprod(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cumprod(int64_t dim)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cumprod(self_, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cumsum(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cumsum(int64_t dim)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_cumsum(self_, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_data_ptr(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_data_ptr(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_det(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_det(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_diag(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diag(int64_t diagonal=0)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_diag(self_, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_digamma(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_digamma(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_digamma_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_digamma_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_dist(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dist(Tensor other, Scalar p=2)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_dist(self_, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_div(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "div(Scalar other)",
    "div(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_div(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_div(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_div_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "div_(Scalar other)",
    "div_(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_div_(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_div_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_dot(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dot(Tensor tensor)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_dot(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_eig(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eig(bool eigenvectors=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_eig(self_, r.toBool(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_eq(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eq(Scalar other)",
    "eq(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_eq(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_eq(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_eq_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eq_(Scalar other)",
    "eq_(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_eq_(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_eq_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_equal(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "equal(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_equal(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erf(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_erf(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erf_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_erf_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erfinv(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_erfinv(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erfinv_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_erfinv_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_exp(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_exp(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_exp_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_exp_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_expand(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "expand(IntList size)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_expand(self_, r.intlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_expand_as(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "expand_as(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_expand_as(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_expm1(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_expm1(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_expm1_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_expm1_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_exponential_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "exponential_(double lambd=1, *, Generator generator=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_exponential_(self_, r.toDouble(0), r.generator(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fill_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fill_(Scalar value)",
    "fill_(Tensor value)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_fill_(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_fill_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_floor(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_floor(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_floor_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_floor_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fmod(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fmod(Scalar other)",
    "fmod(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_fmod(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_fmod(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fmod_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fmod_(Scalar other)",
    "fmod_(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_fmod_(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_fmod_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_frac(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_frac(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_frac_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_frac_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gather(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gather(int64_t dim, Tensor index)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_gather(self_, r.toInt64(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ge(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ge(Scalar other)",
    "ge(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_ge(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_ge(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ge_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ge_(Scalar other)",
    "ge_(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_ge_(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_ge_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gels(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gels(Tensor A)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_gels(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_geometric_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "geometric_(double p, *, Generator generator=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_geometric_(self_, r.toDouble(0), r.generator(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_geqrf(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_geqrf(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ger(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ger(Tensor vec2)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_ger(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gesv(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gesv(Tensor A)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_gesv(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_get_device(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_get_device(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gt(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gt(Scalar other)",
    "gt(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_gt(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_gt(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gt_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gt_(Scalar other)",
    "gt_(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_gt_(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_gt_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_histc(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "histc(int64_t bins=100, Scalar min=0, Scalar max=0)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_histc(self_, r.toInt64(0), r.scalar(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index(TensorList indices)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_index(self_, r.tensorlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_add_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_add_(int64_t dim, Tensor index, Tensor source)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_index_add_(self_, r.toInt64(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_copy_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_copy_(int64_t dim, Tensor index, Tensor source)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_index_copy_(self_, r.toInt64(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_fill_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_fill_(int64_t dim, Tensor index, Scalar value)",
    "index_fill_(int64_t dim, Tensor index, Tensor value)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_index_fill_(self_, r.toInt64(0), r.tensor(1), r.scalar(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_index_fill_(self_, r.toInt64(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_put_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_put_(TensorList indices, Tensor values)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_index_put_(self_, r.tensorlist(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_select(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_select(int64_t dim, Tensor index)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_index_select(self_, r.toInt64(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_inverse(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_inverse(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_coalesced(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_is_coalesced(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_contiguous(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_is_contiguous(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_distributed(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_is_distributed(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_floating_point(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_is_floating_point(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_nonzero(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_is_nonzero(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_same_size(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_same_size(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_is_same_size(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_set_to(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_set_to(Tensor tensor)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_is_set_to(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_signed(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_is_signed(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_kthvalue(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "kthvalue(int64_t k, int64_t dim=-1, bool keepdim=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_kthvalue(self_, r.toInt64(0), r.toInt64(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_le(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "le(Scalar other)",
    "le(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_le(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_le(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_le_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "le_(Scalar other)",
    "le_(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_le_(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_le_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lerp(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lerp(Tensor end, Scalar weight)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_lerp(self_, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lerp_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lerp_(Tensor end, Scalar weight)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_lerp_(self_, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lgamma(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_lgamma(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lgamma_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_lgamma_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_log(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log1p(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_log1p(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log1p_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_log1p_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_log_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log_normal_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log_normal_(double mean=1, double std=2, *, Generator generator=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_log_normal_(self_, r.toDouble(0), r.toDouble(1), r.generator(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lt(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lt(Scalar other)",
    "lt(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_lt(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_lt(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lt_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lt_(Scalar other)",
    "lt_(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_lt_(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_lt_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_masked_fill_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_fill_(Tensor mask, Scalar value)",
    "masked_fill_(Tensor mask, Tensor value)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_masked_fill_(self_, r.tensor(0), r.scalar(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_masked_fill_(self_, r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_masked_scatter_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_scatter_(Tensor mask, Tensor source)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_masked_scatter_(self_, r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_masked_select(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_select(Tensor mask)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_masked_select(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_matmul(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "matmul(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_matmul(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_max(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max()",
    "max(Tensor other)",
    "max(int64_t dim, bool keepdim=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_max(self_));
  } else if (r.idx == 1) {
    return wrap(dispatch_max(self_, r.tensor(0)));
  } else if (r.idx == 2) {
    return wrap(dispatch_max(self_, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mean(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mean()",
    "mean(int64_t dim, bool keepdim=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_mean(self_));
  } else if (r.idx == 1) {
    return wrap(dispatch_mean(self_, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_median(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "median()",
    "median(int64_t dim, bool keepdim=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_median(self_));
  } else if (r.idx == 1) {
    return wrap(dispatch_median(self_, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_min(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "min()",
    "min(Tensor other)",
    "min(int64_t dim, bool keepdim=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_min(self_));
  } else if (r.idx == 1) {
    return wrap(dispatch_min(self_, r.tensor(0)));
  } else if (r.idx == 2) {
    return wrap(dispatch_min(self_, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mm(Tensor mat2)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_mm(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mode(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mode(int64_t dim=-1, bool keepdim=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_mode(self_, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mul(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mul(Scalar other)",
    "mul(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_mul(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_mul(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mul_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mul_(Scalar other)",
    "mul_(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_mul_(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_mul_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_multinomial(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "multinomial(int64_t num_samples, bool replacement=False, *, Generator generator=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_multinomial(self_, r.toInt64(0), r.toBool(1), r.generator(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mv(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mv(Tensor vec)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_mv(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_narrow(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "narrow(int64_t dim, int64_t start, int64_t length)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_narrow(self_, r.toInt64(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ne(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ne(Scalar other)",
    "ne(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_ne(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_ne(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ne_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ne_(Scalar other)",
    "ne_(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_ne_(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_ne_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_neg(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_neg(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_neg_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_neg_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_nonzero(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_nonzero(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_norm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "norm(Scalar p=2)",
    "norm(Scalar p=None, int64_t dim, bool keepdim=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_norm(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    auto p = r.scalarWithDefault(0, 2);
    auto dim = r.toInt64(1);
    auto keepdim = r.toBool(2);
    return wrap(dispatch_norm(self_, p, dim, keepdim));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_normal_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "normal_(double mean=0, double std=1, *, Generator generator=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_normal_(self_, r.toDouble(0), r.toDouble(1), r.generator(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_numel(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_numel(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_orgqr(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "orgqr(Tensor input2)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_orgqr(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ormqr(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ormqr(Tensor input2, Tensor input3, bool left=True, bool transpose=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_ormqr(self_, r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_permute(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "permute(IntList dims)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_permute(self_, r.intlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pin_memory(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_pin_memory(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_polygamma(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "polygamma(int64_t n, )",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_polygamma(r.toInt64(0), self_));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_polygamma_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "polygamma_(int64_t n)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_polygamma_(self_, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_potrf(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "potrf(bool upper=True)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_potrf(self_, r.toBool(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_potri(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "potri(bool upper=True)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_potri(self_, r.toBool(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_potrs(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "potrs(Tensor input2, bool upper=True)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_potrs(self_, r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pow(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pow(Scalar exponent)",
    "pow(Tensor exponent)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_pow(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_pow(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pow_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pow_(Scalar exponent)",
    "pow_(Tensor exponent)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_pow_(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_pow_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_prod(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "prod()",
    "prod(int64_t dim, bool keepdim=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_prod(self_));
  } else if (r.idx == 1) {
    return wrap(dispatch_prod(self_, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pstrf(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pstrf(bool upper=True, Scalar tol=-1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_pstrf(self_, r.toBool(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_put_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "put_(Tensor index, Tensor source, bool accumulate=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_put_(self_, r.tensor(0), r.tensor(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_qr(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_qr(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_random_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "random_(*, Generator generator=None)",
    "random_(int64_t from, int64_t to, *, Generator generator=None)",
    "random_(int64_t to, *, Generator generator=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_random_(self_, r.generator(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_random_(self_, r.toInt64(0), r.toInt64(1), r.generator(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_random_(self_, r.toInt64(0), r.generator(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_reciprocal(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_reciprocal(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_reciprocal_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_reciprocal_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_remainder(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "remainder(Scalar other)",
    "remainder(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_remainder(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_remainder(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_remainder_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "remainder_(Scalar other)",
    "remainder_(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_remainder_(self_, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_remainder_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_renorm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "renorm(Scalar p, int64_t dim, Scalar maxnorm)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_renorm(self_, r.scalar(0), r.toInt64(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_renorm_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "renorm_(Scalar p, int64_t dim, Scalar maxnorm)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_renorm_(self_, r.scalar(0), r.toInt64(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_repeat(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "repeat(IntList repeats)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_repeat(self_, r.intlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_reshape_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reshape_(IntList size, IntList stride)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_reshape_(self_, r.intlist(0), r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_resize_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "resize_(IntList size)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_resize_(self_, r.intlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_resize_as_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "resize_as_(Tensor the_template)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_resize_as_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_round(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_round(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_round_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_round_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rsqrt(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_rsqrt(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rsqrt_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_rsqrt_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_scatter_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scatter_(int64_t dim, Tensor index, Scalar value)",
    "scatter_(int64_t dim, Tensor index, Tensor src)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_scatter_(self_, r.toInt64(0), r.tensor(1), r.scalar(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_scatter_(self_, r.toInt64(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_scatter_add_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scatter_add_(int64_t dim, Tensor index, Tensor src)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_scatter_add_(self_, r.toInt64(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_select(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "select(int64_t dim, int64_t index)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_select(self_, r.toInt64(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_set_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "set_()",
    "set_(Storage sourceStorage, int64_t storage_offset, IntList size, IntList stride=None)",
    "set_(Storage storage)",
    "set_(Tensor source)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_set_(self_));
  } else if (r.idx == 1) {
    return wrap(dispatch_set_(self_, *r.storage(0), r.toInt64(1), r.intlist(2), r.intlist(3)));
  } else if (r.idx == 2) {
    return wrap(dispatch_set_(self_, *r.storage(0)));
  } else if (r.idx == 3) {
    return wrap(dispatch_set_(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sigmoid(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_sigmoid(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sigmoid_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_sigmoid_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sign(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_sign(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sign_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_sign_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sin(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_sin(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sin_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_sin_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sinh(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_sinh(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sinh_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_sinh_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_slice(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slice(int64_t dim=0, int64_t start=0, int64_t end=9223372036854775807, int64_t step=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_slice(self_, r.toInt64(0), r.toInt64(1), r.toInt64(2), r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_smm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "smm(Tensor mat2)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_smm(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sort(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sort(int64_t dim=-1, bool descending=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sort(self_, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_split(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "split(int64_t split_size, int64_t dim=0)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_split(self_, r.toInt64(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sqrt(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_sqrt(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sqrt_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_sqrt_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_squeeze(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "squeeze()",
    "squeeze(int64_t dim)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_squeeze(self_));
  } else if (r.idx == 1) {
    return wrap(dispatch_squeeze(self_, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_squeeze_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "squeeze_()",
    "squeeze_(int64_t dim)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_squeeze_(self_));
  } else if (r.idx == 1) {
    return wrap(dispatch_squeeze_(self_, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sspaddmm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sspaddmm(Scalar beta, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Scalar beta, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sspaddmm(r.scalar(0), self_, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_sspaddmm(r.scalar(0), self_, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_sspaddmm(self_, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_std(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "std(bool unbiased=True)",
    "std(int64_t dim, bool unbiased=True, bool keepdim=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_std(self_, r.toBool(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_std(self_, r.toInt64(0), r.toBool(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_stft(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stft(int64_t frame_length, int64_t hop, int64_t fft_size=None, bool return_onesided=True, Tensor window=None, int64_t pad_end=0)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[7];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto frame_length = r.toInt64(0);
    auto hop = r.toInt64(1);
    auto fft_size = r.toInt64WithDefault(2, frame_length);
    auto return_onesided = r.toBool(3);
    auto window = r.tensor(4);
    auto pad_end = r.toInt64(5);
    return wrap(dispatch_stft(self_, frame_length, hop, fft_size, return_onesided, window, pad_end));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_storage_offset(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_storage_offset(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sub(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sub(Scalar alpha, Tensor other)|deprecated",
    "sub(Scalar other, *, Scalar alpha=1)",
    "sub(Tensor other, *, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sub(self_, r.scalar(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_sub(self_, r.scalar(0), r.scalar(1)));
  } else if (r.idx == 2) {
    return wrap(dispatch_sub(self_, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sub_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sub_(Scalar alpha, Tensor other)|deprecated",
    "sub_(Scalar other, *, Scalar alpha=1)",
    "sub_(Tensor other, *, Scalar alpha=1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sub_(self_, r.scalar(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_sub_(self_, r.scalar(0), r.scalar(1)));
  } else if (r.idx == 2) {
    return wrap(dispatch_sub_(self_, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sum(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sum()",
    "sum(int64_t dim, bool keepdim=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_sum(self_));
  } else if (r.idx == 1) {
    return wrap(dispatch_sum(self_, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_svd(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "svd(bool some=True)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_svd(self_, r.toBool(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_symeig(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "symeig(bool eigenvectors=False, bool upper=True)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_symeig(self_, r.toBool(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_t(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_t(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_t_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_t_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_take(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "take(Tensor index)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_take(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tan(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_tan(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tan_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_tan_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tanh(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_tanh(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tanh_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_tanh_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_to_dense(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_to_dense(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_topk(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "topk(int64_t k, int64_t dim=-1, bool largest=True, bool sorted=True)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_topk(self_, r.toInt64(0), r.toInt64(1), r.toBool(2), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trace(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_trace(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_transpose(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "transpose(int64_t dim0, int64_t dim1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_transpose(self_, r.toInt64(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_transpose_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "transpose_(int64_t dim0, int64_t dim1)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_transpose_(self_, r.toInt64(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tril(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tril(int64_t diagonal=0)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_tril(self_, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tril_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tril_(int64_t diagonal=0)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_tril_(self_, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_triu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triu(int64_t diagonal=0)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_triu(self_, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_triu_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triu_(int64_t diagonal=0)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_triu_(self_, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trtrs(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trtrs(Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[5];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_trtrs(self_, r.tensor(0), r.toBool(1), r.toBool(2), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trunc(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_trunc(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trunc_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_trunc_(self_));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_type_as(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "type_as(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_type_as(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_unfold(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unfold(int64_t dimension, int64_t size, int64_t step)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_unfold(self_, r.toInt64(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_uniform_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "uniform_(double from=0, double to=1, *, Generator generator=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_uniform_(self_, r.toDouble(0), r.toDouble(1), r.generator(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_unsqueeze(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unsqueeze(int64_t dim)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_unsqueeze(self_, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_unsqueeze_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unsqueeze_(int64_t dim)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_unsqueeze_(self_, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_var(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "var(bool unbiased=True)",
    "var(int64_t dim, bool unbiased=True, bool keepdim=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[4];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_var(self_, r.toBool(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_var(self_, r.toInt64(0), r.toBool(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_view(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "view(IntList size)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_view(self_, r.intlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_view_as(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "view_as(Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_view_as(self_, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_where(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "where(Tensor condition, Tensor other)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[3];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_where(r.tensor(0), self_, r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_zero_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_zero_(self_));
  END_HANDLE_TH_ERRORS
}

PyMethodDef variable_methods[] = {
  {"__add__", (PyCFunction)THPVariable_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__radd__", (PyCFunction)THPVariable_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__iadd__", (PyCFunction)THPVariable_add_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__rmul__", (PyCFunction)THPVariable_mul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mul__", (PyCFunction)THPVariable_mul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__imul__", (PyCFunction)THPVariable_mul_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__sub__", (PyCFunction)THPVariable_sub, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__isub__", (PyCFunction)THPVariable_sub_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__div__", (PyCFunction)THPVariable_div, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__truediv__", (PyCFunction)THPVariable_div, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__idiv__", (PyCFunction)THPVariable_div_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mod__", (PyCFunction)THPVariable_remainder, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__bool__", (PyCFunction)THPVariable_is_nonzero, METH_NOARGS, NULL},
  {"__float__", (PyCFunction)THPVariable_float_scalar, METH_NOARGS, NULL},
  {"__int__", (PyCFunction)THPVariable_integral_scalar, METH_NOARGS, NULL},
  {"__long__", (PyCFunction)THPVariable_integral_scalar, METH_NOARGS, NULL},
  {"__index__", (PyCFunction)THPVariable_index_scalar, METH_NOARGS, NULL},
  {"__invert__", (PyCFunction)THPVariable_invert, METH_NOARGS, NULL},
  {"__nonzero__", (PyCFunction)THPVariable_is_nonzero, METH_NOARGS, NULL},
  {"__matmul__", (PyCFunction)THPVariable_matmul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"apply_", (PyCFunction)THPVariable_apply_, METH_O, NULL},
  {"byte", (PyCFunction)THPVariable_byte, METH_NOARGS, NULL},
  {"char", (PyCFunction)THPVariable_char, METH_NOARGS, NULL},
  {"clamp", (PyCFunction)THPVariable_clamp, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp_", (PyCFunction)THPVariable_clamp_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"contiguous", (PyCFunction)THPVariable_contiguous, METH_NOARGS, NULL},
  {"copy_", (PyCFunction)THPVariable_copy_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cpu", (PyCFunction)THPVariable_cpu, METH_NOARGS, NULL},
  {"cuda", (PyCFunction)THPVariable_cuda, METH_VARARGS | METH_KEYWORDS, NULL},
  {"dim", (PyCFunction)THPVariable_dim, METH_NOARGS, NULL},
  {"detach", (PyCFunction)THPVariable_detach, METH_NOARGS, NULL},
  {"detach_", (PyCFunction)THPVariable_detach_, METH_NOARGS, NULL},
  {"double", (PyCFunction)THPVariable_double, METH_NOARGS, NULL},
  {"element_size", (PyCFunction)THPVariable_element_size, METH_NOARGS, NULL},
  {"float", (PyCFunction)THPVariable_float, METH_NOARGS, NULL},
  {"half", (PyCFunction)THPVariable_half, METH_NOARGS, NULL},
  {"int", (PyCFunction)THPVariable_int, METH_NOARGS, NULL},
  {"item", (PyCFunction)THPVariable_item, METH_NOARGS, NULL},
  {"long", (PyCFunction)THPVariable_long, METH_NOARGS, NULL},
  {"map_", (PyCFunction)THPVariable_map_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"map2_", (PyCFunction)THPVariable_map2_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ndimension", (PyCFunction)THPVariable_dim, METH_NOARGS, NULL},
  {"nelement", (PyCFunction)THPVariable_numel, METH_NOARGS, NULL},
  {"new", (PyCFunction)THPVariable_new, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_tensor", (PyCFunction)THPVariable_new_tensor, METH_VARARGS | METH_KEYWORDS, NULL},
  {"numpy", (PyCFunction)THPVariable_numpy, METH_NOARGS, NULL},
  {"record_stream", (PyCFunction)THPVariable_record_stream, METH_O, NULL},
  {"short", (PyCFunction)THPVariable_short, METH_NOARGS, NULL},
  {"size", (PyCFunction)THPVariable_size, METH_VARARGS | METH_KEYWORDS, NULL},
  {"storage", (PyCFunction)THPVariable_storage, METH_NOARGS, NULL},
  {"storage_type", (PyCFunction)THPVariable_storage_type, METH_NOARGS, NULL},
  {"stride", (PyCFunction)THPVariable_stride, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tolist", (PyCFunction)THPVariable_tolist, METH_NOARGS, NULL},
  {"type", (PyCFunction)THPVariable_type, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__and__", (PyCFunction)THPVariable___and__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__iand__", (PyCFunction)THPVariable___iand__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__ilshift__", (PyCFunction)THPVariable___ilshift__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__ior__", (PyCFunction)THPVariable___ior__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__irshift__", (PyCFunction)THPVariable___irshift__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__ixor__", (PyCFunction)THPVariable___ixor__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__lshift__", (PyCFunction)THPVariable___lshift__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__or__", (PyCFunction)THPVariable___or__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__rshift__", (PyCFunction)THPVariable___rshift__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__xor__", (PyCFunction)THPVariable___xor__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_addmv", (PyCFunction)THPVariable__addmv, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_addmv_", (PyCFunction)THPVariable__addmv_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_addr", (PyCFunction)THPVariable__addr, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_addr_", (PyCFunction)THPVariable__addr_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_copy_ignoring_overlaps_", (PyCFunction)THPVariable__copy_ignoring_overlaps_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_det_with_svd", (PyCFunction)THPVariable__det_with_svd, METH_NOARGS, NULL},
  {"_dimI", (PyCFunction)THPVariable__dimI, METH_NOARGS, NULL},
  {"_dimV", (PyCFunction)THPVariable__dimV, METH_NOARGS, NULL},
  {"_dot", (PyCFunction)THPVariable__dot, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_ger", (PyCFunction)THPVariable__ger, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_indices", (PyCFunction)THPVariable__indices, METH_NOARGS, NULL},
  {"_mm", (PyCFunction)THPVariable__mm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_mv", (PyCFunction)THPVariable__mv, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_nnz", (PyCFunction)THPVariable__nnz, METH_NOARGS, NULL},
  {"_s_where", (PyCFunction)THPVariable__s_where, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_sparse_mask", (PyCFunction)THPVariable__sparse_mask, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_standard_gamma", (PyCFunction)THPVariable__standard_gamma, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_standard_gamma_grad", (PyCFunction)THPVariable__standard_gamma_grad, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_values", (PyCFunction)THPVariable__values, METH_NOARGS, NULL},
  {"abs", (PyCFunction)THPVariable_abs, METH_NOARGS, NULL},
  {"abs_", (PyCFunction)THPVariable_abs_, METH_NOARGS, NULL},
  {"acos", (PyCFunction)THPVariable_acos, METH_NOARGS, NULL},
  {"acos_", (PyCFunction)THPVariable_acos_, METH_NOARGS, NULL},
  {"add", (PyCFunction)THPVariable_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"add_", (PyCFunction)THPVariable_add_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addbmm", (PyCFunction)THPVariable_addbmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addbmm_", (PyCFunction)THPVariable_addbmm_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcdiv", (PyCFunction)THPVariable_addcdiv, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcdiv_", (PyCFunction)THPVariable_addcdiv_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcmul", (PyCFunction)THPVariable_addcmul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcmul_", (PyCFunction)THPVariable_addcmul_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmm", (PyCFunction)THPVariable_addmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmm_", (PyCFunction)THPVariable_addmm_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmv", (PyCFunction)THPVariable_addmv, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmv_", (PyCFunction)THPVariable_addmv_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addr", (PyCFunction)THPVariable_addr, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addr_", (PyCFunction)THPVariable_addr_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"all", (PyCFunction)THPVariable_all, METH_NOARGS, NULL},
  {"allclose", (PyCFunction)THPVariable_allclose, METH_VARARGS | METH_KEYWORDS, NULL},
  {"any", (PyCFunction)THPVariable_any, METH_NOARGS, NULL},
  {"as_strided", (PyCFunction)THPVariable_as_strided, METH_VARARGS | METH_KEYWORDS, NULL},
  {"as_strided_", (PyCFunction)THPVariable_as_strided_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"asin", (PyCFunction)THPVariable_asin, METH_NOARGS, NULL},
  {"asin_", (PyCFunction)THPVariable_asin_, METH_NOARGS, NULL},
  {"atan", (PyCFunction)THPVariable_atan, METH_NOARGS, NULL},
  {"atan2", (PyCFunction)THPVariable_atan2, METH_VARARGS | METH_KEYWORDS, NULL},
  {"atan2_", (PyCFunction)THPVariable_atan2_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"atan_", (PyCFunction)THPVariable_atan_, METH_NOARGS, NULL},
  {"baddbmm", (PyCFunction)THPVariable_baddbmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"baddbmm_", (PyCFunction)THPVariable_baddbmm_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bernoulli", (PyCFunction)THPVariable_bernoulli, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bernoulli_", (PyCFunction)THPVariable_bernoulli_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bmm", (PyCFunction)THPVariable_bmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"btrifact", (PyCFunction)THPVariable_btrifact, METH_VARARGS | METH_KEYWORDS, NULL},
  {"btrifact_with_info", (PyCFunction)THPVariable_btrifact_with_info, METH_VARARGS | METH_KEYWORDS, NULL},
  {"btrisolve", (PyCFunction)THPVariable_btrisolve, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cauchy_", (PyCFunction)THPVariable_cauchy_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ceil", (PyCFunction)THPVariable_ceil, METH_NOARGS, NULL},
  {"ceil_", (PyCFunction)THPVariable_ceil_, METH_NOARGS, NULL},
  {"chunk", (PyCFunction)THPVariable_chunk, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clone", (PyCFunction)THPVariable_clone, METH_NOARGS, NULL},
  {"coalesce", (PyCFunction)THPVariable_coalesce, METH_NOARGS, NULL},
  {"conv_tbc", (PyCFunction)THPVariable_conv_tbc, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cos", (PyCFunction)THPVariable_cos, METH_NOARGS, NULL},
  {"cos_", (PyCFunction)THPVariable_cos_, METH_NOARGS, NULL},
  {"cosh", (PyCFunction)THPVariable_cosh, METH_NOARGS, NULL},
  {"cosh_", (PyCFunction)THPVariable_cosh_, METH_NOARGS, NULL},
  {"cross", (PyCFunction)THPVariable_cross, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cumprod", (PyCFunction)THPVariable_cumprod, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cumsum", (PyCFunction)THPVariable_cumsum, METH_VARARGS | METH_KEYWORDS, NULL},
  {"data_ptr", (PyCFunction)THPVariable_data_ptr, METH_NOARGS, NULL},
  {"det", (PyCFunction)THPVariable_det, METH_NOARGS, NULL},
  {"diag", (PyCFunction)THPVariable_diag, METH_VARARGS | METH_KEYWORDS, NULL},
  {"digamma", (PyCFunction)THPVariable_digamma, METH_NOARGS, NULL},
  {"digamma_", (PyCFunction)THPVariable_digamma_, METH_NOARGS, NULL},
  {"dist", (PyCFunction)THPVariable_dist, METH_VARARGS | METH_KEYWORDS, NULL},
  {"div", (PyCFunction)THPVariable_div, METH_VARARGS | METH_KEYWORDS, NULL},
  {"div_", (PyCFunction)THPVariable_div_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"dot", (PyCFunction)THPVariable_dot, METH_VARARGS | METH_KEYWORDS, NULL},
  {"eig", (PyCFunction)THPVariable_eig, METH_VARARGS | METH_KEYWORDS, NULL},
  {"eq", (PyCFunction)THPVariable_eq, METH_VARARGS | METH_KEYWORDS, NULL},
  {"eq_", (PyCFunction)THPVariable_eq_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"equal", (PyCFunction)THPVariable_equal, METH_VARARGS | METH_KEYWORDS, NULL},
  {"erf", (PyCFunction)THPVariable_erf, METH_NOARGS, NULL},
  {"erf_", (PyCFunction)THPVariable_erf_, METH_NOARGS, NULL},
  {"erfinv", (PyCFunction)THPVariable_erfinv, METH_NOARGS, NULL},
  {"erfinv_", (PyCFunction)THPVariable_erfinv_, METH_NOARGS, NULL},
  {"exp", (PyCFunction)THPVariable_exp, METH_NOARGS, NULL},
  {"exp_", (PyCFunction)THPVariable_exp_, METH_NOARGS, NULL},
  {"expand", (PyCFunction)THPVariable_expand, METH_VARARGS | METH_KEYWORDS, NULL},
  {"expand_as", (PyCFunction)THPVariable_expand_as, METH_VARARGS | METH_KEYWORDS, NULL},
  {"expm1", (PyCFunction)THPVariable_expm1, METH_NOARGS, NULL},
  {"expm1_", (PyCFunction)THPVariable_expm1_, METH_NOARGS, NULL},
  {"exponential_", (PyCFunction)THPVariable_exponential_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"fill_", (PyCFunction)THPVariable_fill_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"floor", (PyCFunction)THPVariable_floor, METH_NOARGS, NULL},
  {"floor_", (PyCFunction)THPVariable_floor_, METH_NOARGS, NULL},
  {"fmod", (PyCFunction)THPVariable_fmod, METH_VARARGS | METH_KEYWORDS, NULL},
  {"fmod_", (PyCFunction)THPVariable_fmod_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"frac", (PyCFunction)THPVariable_frac, METH_NOARGS, NULL},
  {"frac_", (PyCFunction)THPVariable_frac_, METH_NOARGS, NULL},
  {"gather", (PyCFunction)THPVariable_gather, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ge", (PyCFunction)THPVariable_ge, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ge_", (PyCFunction)THPVariable_ge_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"gels", (PyCFunction)THPVariable_gels, METH_VARARGS | METH_KEYWORDS, NULL},
  {"geometric_", (PyCFunction)THPVariable_geometric_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"geqrf", (PyCFunction)THPVariable_geqrf, METH_NOARGS, NULL},
  {"ger", (PyCFunction)THPVariable_ger, METH_VARARGS | METH_KEYWORDS, NULL},
  {"gesv", (PyCFunction)THPVariable_gesv, METH_VARARGS | METH_KEYWORDS, NULL},
  {"get_device", (PyCFunction)THPVariable_get_device, METH_NOARGS, NULL},
  {"gt", (PyCFunction)THPVariable_gt, METH_VARARGS | METH_KEYWORDS, NULL},
  {"gt_", (PyCFunction)THPVariable_gt_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"histc", (PyCFunction)THPVariable_histc, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index", (PyCFunction)THPVariable_index, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_add_", (PyCFunction)THPVariable_index_add_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_copy_", (PyCFunction)THPVariable_index_copy_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_fill_", (PyCFunction)THPVariable_index_fill_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_put_", (PyCFunction)THPVariable_index_put_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_select", (PyCFunction)THPVariable_index_select, METH_VARARGS | METH_KEYWORDS, NULL},
  {"inverse", (PyCFunction)THPVariable_inverse, METH_NOARGS, NULL},
  {"is_coalesced", (PyCFunction)THPVariable_is_coalesced, METH_NOARGS, NULL},
  {"is_contiguous", (PyCFunction)THPVariable_is_contiguous, METH_NOARGS, NULL},
  {"is_distributed", (PyCFunction)THPVariable_is_distributed, METH_NOARGS, NULL},
  {"is_floating_point", (PyCFunction)THPVariable_is_floating_point, METH_NOARGS, NULL},
  {"is_nonzero", (PyCFunction)THPVariable_is_nonzero, METH_NOARGS, NULL},
  {"is_same_size", (PyCFunction)THPVariable_is_same_size, METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_set_to", (PyCFunction)THPVariable_is_set_to, METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_signed", (PyCFunction)THPVariable_is_signed, METH_NOARGS, NULL},
  {"kthvalue", (PyCFunction)THPVariable_kthvalue, METH_VARARGS | METH_KEYWORDS, NULL},
  {"le", (PyCFunction)THPVariable_le, METH_VARARGS | METH_KEYWORDS, NULL},
  {"le_", (PyCFunction)THPVariable_le_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lerp", (PyCFunction)THPVariable_lerp, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lerp_", (PyCFunction)THPVariable_lerp_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lgamma", (PyCFunction)THPVariable_lgamma, METH_NOARGS, NULL},
  {"lgamma_", (PyCFunction)THPVariable_lgamma_, METH_NOARGS, NULL},
  {"log", (PyCFunction)THPVariable_log, METH_NOARGS, NULL},
  {"log1p", (PyCFunction)THPVariable_log1p, METH_NOARGS, NULL},
  {"log1p_", (PyCFunction)THPVariable_log1p_, METH_NOARGS, NULL},
  {"log_", (PyCFunction)THPVariable_log_, METH_NOARGS, NULL},
  {"log_normal_", (PyCFunction)THPVariable_log_normal_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lt", (PyCFunction)THPVariable_lt, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lt_", (PyCFunction)THPVariable_lt_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"masked_fill_", (PyCFunction)THPVariable_masked_fill_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"masked_scatter_", (PyCFunction)THPVariable_masked_scatter_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"masked_select", (PyCFunction)THPVariable_masked_select, METH_VARARGS | METH_KEYWORDS, NULL},
  {"matmul", (PyCFunction)THPVariable_matmul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"max", (PyCFunction)THPVariable_max, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mean", (PyCFunction)THPVariable_mean, METH_VARARGS | METH_KEYWORDS, NULL},
  {"median", (PyCFunction)THPVariable_median, METH_VARARGS | METH_KEYWORDS, NULL},
  {"min", (PyCFunction)THPVariable_min, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mm", (PyCFunction)THPVariable_mm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mode", (PyCFunction)THPVariable_mode, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mul", (PyCFunction)THPVariable_mul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mul_", (PyCFunction)THPVariable_mul_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"multinomial", (PyCFunction)THPVariable_multinomial, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mv", (PyCFunction)THPVariable_mv, METH_VARARGS | METH_KEYWORDS, NULL},
  {"narrow", (PyCFunction)THPVariable_narrow, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ne", (PyCFunction)THPVariable_ne, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ne_", (PyCFunction)THPVariable_ne_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"neg", (PyCFunction)THPVariable_neg, METH_NOARGS, NULL},
  {"neg_", (PyCFunction)THPVariable_neg_, METH_NOARGS, NULL},
  {"nonzero", (PyCFunction)THPVariable_nonzero, METH_NOARGS, NULL},
  {"norm", (PyCFunction)THPVariable_norm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"normal_", (PyCFunction)THPVariable_normal_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"numel", (PyCFunction)THPVariable_numel, METH_NOARGS, NULL},
  {"orgqr", (PyCFunction)THPVariable_orgqr, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ormqr", (PyCFunction)THPVariable_ormqr, METH_VARARGS | METH_KEYWORDS, NULL},
  {"permute", (PyCFunction)THPVariable_permute, METH_VARARGS | METH_KEYWORDS, NULL},
  {"pin_memory", (PyCFunction)THPVariable_pin_memory, METH_NOARGS, NULL},
  {"polygamma", (PyCFunction)THPVariable_polygamma, METH_VARARGS | METH_KEYWORDS, NULL},
  {"polygamma_", (PyCFunction)THPVariable_polygamma_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"potrf", (PyCFunction)THPVariable_potrf, METH_VARARGS | METH_KEYWORDS, NULL},
  {"potri", (PyCFunction)THPVariable_potri, METH_VARARGS | METH_KEYWORDS, NULL},
  {"potrs", (PyCFunction)THPVariable_potrs, METH_VARARGS | METH_KEYWORDS, NULL},
  {"pow", (PyCFunction)THPVariable_pow, METH_VARARGS | METH_KEYWORDS, NULL},
  {"pow_", (PyCFunction)THPVariable_pow_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"prod", (PyCFunction)THPVariable_prod, METH_VARARGS | METH_KEYWORDS, NULL},
  {"pstrf", (PyCFunction)THPVariable_pstrf, METH_VARARGS | METH_KEYWORDS, NULL},
  {"put_", (PyCFunction)THPVariable_put_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"qr", (PyCFunction)THPVariable_qr, METH_NOARGS, NULL},
  {"random_", (PyCFunction)THPVariable_random_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"reciprocal", (PyCFunction)THPVariable_reciprocal, METH_NOARGS, NULL},
  {"reciprocal_", (PyCFunction)THPVariable_reciprocal_, METH_NOARGS, NULL},
  {"remainder", (PyCFunction)THPVariable_remainder, METH_VARARGS | METH_KEYWORDS, NULL},
  {"remainder_", (PyCFunction)THPVariable_remainder_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"renorm", (PyCFunction)THPVariable_renorm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"renorm_", (PyCFunction)THPVariable_renorm_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"repeat", (PyCFunction)THPVariable_repeat, METH_VARARGS | METH_KEYWORDS, NULL},
  {"reshape_", (PyCFunction)THPVariable_reshape_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"resize_", (PyCFunction)THPVariable_resize_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"resize_as_", (PyCFunction)THPVariable_resize_as_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"round", (PyCFunction)THPVariable_round, METH_NOARGS, NULL},
  {"round_", (PyCFunction)THPVariable_round_, METH_NOARGS, NULL},
  {"rsqrt", (PyCFunction)THPVariable_rsqrt, METH_NOARGS, NULL},
  {"rsqrt_", (PyCFunction)THPVariable_rsqrt_, METH_NOARGS, NULL},
  {"scatter_", (PyCFunction)THPVariable_scatter_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"scatter_add_", (PyCFunction)THPVariable_scatter_add_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"select", (PyCFunction)THPVariable_select, METH_VARARGS | METH_KEYWORDS, NULL},
  {"set_", (PyCFunction)THPVariable_set_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sigmoid", (PyCFunction)THPVariable_sigmoid, METH_NOARGS, NULL},
  {"sigmoid_", (PyCFunction)THPVariable_sigmoid_, METH_NOARGS, NULL},
  {"sign", (PyCFunction)THPVariable_sign, METH_NOARGS, NULL},
  {"sign_", (PyCFunction)THPVariable_sign_, METH_NOARGS, NULL},
  {"sin", (PyCFunction)THPVariable_sin, METH_NOARGS, NULL},
  {"sin_", (PyCFunction)THPVariable_sin_, METH_NOARGS, NULL},
  {"sinh", (PyCFunction)THPVariable_sinh, METH_NOARGS, NULL},
  {"sinh_", (PyCFunction)THPVariable_sinh_, METH_NOARGS, NULL},
  {"slice", (PyCFunction)THPVariable_slice, METH_VARARGS | METH_KEYWORDS, NULL},
  {"smm", (PyCFunction)THPVariable_smm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sort", (PyCFunction)THPVariable_sort, METH_VARARGS | METH_KEYWORDS, NULL},
  {"split", (PyCFunction)THPVariable_split, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sqrt", (PyCFunction)THPVariable_sqrt, METH_NOARGS, NULL},
  {"sqrt_", (PyCFunction)THPVariable_sqrt_, METH_NOARGS, NULL},
  {"squeeze", (PyCFunction)THPVariable_squeeze, METH_VARARGS | METH_KEYWORDS, NULL},
  {"squeeze_", (PyCFunction)THPVariable_squeeze_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sspaddmm", (PyCFunction)THPVariable_sspaddmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"std", (PyCFunction)THPVariable_std, METH_VARARGS | METH_KEYWORDS, NULL},
  {"stft", (PyCFunction)THPVariable_stft, METH_VARARGS | METH_KEYWORDS, NULL},
  {"storage_offset", (PyCFunction)THPVariable_storage_offset, METH_NOARGS, NULL},
  {"sub", (PyCFunction)THPVariable_sub, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sub_", (PyCFunction)THPVariable_sub_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sum", (PyCFunction)THPVariable_sum, METH_VARARGS | METH_KEYWORDS, NULL},
  {"svd", (PyCFunction)THPVariable_svd, METH_VARARGS | METH_KEYWORDS, NULL},
  {"symeig", (PyCFunction)THPVariable_symeig, METH_VARARGS | METH_KEYWORDS, NULL},
  {"t", (PyCFunction)THPVariable_t, METH_NOARGS, NULL},
  {"t_", (PyCFunction)THPVariable_t_, METH_NOARGS, NULL},
  {"take", (PyCFunction)THPVariable_take, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tan", (PyCFunction)THPVariable_tan, METH_NOARGS, NULL},
  {"tan_", (PyCFunction)THPVariable_tan_, METH_NOARGS, NULL},
  {"tanh", (PyCFunction)THPVariable_tanh, METH_NOARGS, NULL},
  {"tanh_", (PyCFunction)THPVariable_tanh_, METH_NOARGS, NULL},
  {"to_dense", (PyCFunction)THPVariable_to_dense, METH_NOARGS, NULL},
  {"topk", (PyCFunction)THPVariable_topk, METH_VARARGS | METH_KEYWORDS, NULL},
  {"trace", (PyCFunction)THPVariable_trace, METH_NOARGS, NULL},
  {"transpose", (PyCFunction)THPVariable_transpose, METH_VARARGS | METH_KEYWORDS, NULL},
  {"transpose_", (PyCFunction)THPVariable_transpose_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tril", (PyCFunction)THPVariable_tril, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tril_", (PyCFunction)THPVariable_tril_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"triu", (PyCFunction)THPVariable_triu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"triu_", (PyCFunction)THPVariable_triu_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"trtrs", (PyCFunction)THPVariable_trtrs, METH_VARARGS | METH_KEYWORDS, NULL},
  {"trunc", (PyCFunction)THPVariable_trunc, METH_NOARGS, NULL},
  {"trunc_", (PyCFunction)THPVariable_trunc_, METH_NOARGS, NULL},
  {"type_as", (PyCFunction)THPVariable_type_as, METH_VARARGS | METH_KEYWORDS, NULL},
  {"unfold", (PyCFunction)THPVariable_unfold, METH_VARARGS | METH_KEYWORDS, NULL},
  {"uniform_", (PyCFunction)THPVariable_uniform_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"unsqueeze", (PyCFunction)THPVariable_unsqueeze, METH_VARARGS | METH_KEYWORDS, NULL},
  {"unsqueeze_", (PyCFunction)THPVariable_unsqueeze_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"var", (PyCFunction)THPVariable_var, METH_VARARGS | METH_KEYWORDS, NULL},
  {"view", (PyCFunction)THPVariable_view, METH_VARARGS | METH_KEYWORDS, NULL},
  {"view_as", (PyCFunction)THPVariable_view_as, METH_VARARGS | METH_KEYWORDS, NULL},
  {"where", (PyCFunction)THPVariable_where, METH_VARARGS | METH_KEYWORDS, NULL},
  {"zero_", (PyCFunction)THPVariable_zero_, METH_NOARGS, NULL},
  {NULL}
};

}} // namespace torch::autograd
