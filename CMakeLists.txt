cmake_minimum_required(VERSION 3.4)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")
#MESSAGE(STATUS "CMAKE_MODULE_PATH : " ${CMAKE_MODULE_PATH})

project(mcv_torch)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-sign-compare -Wno-unused-function -Wno-uninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes -Wno-deprecated-declarations -D_USE_MATH_DEFINES")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_THP_CORE -DWITH_NUMPY -Wno-write-strings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_SCALARS -march=x86-64 -mtune=generic -fopenmp -O2 -pipe -fwrapv")

# ---[ Using cmake scripts and modules

# ---[ Configuration types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Possible configurations" FORCE)
mark_as_advanced(CMAKE_CONFIGURATION_TYPES)

if(DEFINED CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

# --[ If user doesn't specify build type then assume release
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

MESSAGE(STATUS "This is SOURCE dir " ${CMAKE_SOURCE_DIR})
MESSAGE(STATUS "CMAKE_CXX_FLAGS :  " ${CMAKE_CXX_FLAGS})
MESSAGE(STATUS "CMAKE_BUILD_TYPE : " ${CMAKE_BUILD_TYPE})

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CMAKE_CXX_FLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CMAKE_CXX_FLAGS} -O2 -Wall")

set(mcv_torch_LINKER_LIBS)

include_directories("${CMAKE_SOURCE_DIR}/src")

INCLUDE_DIRECTORIES("/mingw/include")
INCLUDE_DIRECTORIES("/mingw64/include/OpenBLAS/")
INCLUDE_DIRECTORIES("C:/msys64/mingw64/include/OpenBLAS")
LINK_DIRECTORIES("/mingw/lib") 

set(PYTHON_EXECUTABLE     "C:/msys64/mingw64/bin/python3.6.exe")
set(PYTHON_INCLUDE_DIR    "C:/msys64/mingw64/include/python3.6m")
set(PYTHON_LIBRARY        "C:/msys64/mingw64/lib/libpython3.6.dll.a")

include(ExternalProject)

include(cmake/External/aten.cmake)
include(cmake/External/shm.cmake)
include(cmake/External/nanopb.cmake)

INCLUDE_DIRECTORIES("${PYTHON_INCLUDE_DIR}")

INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${CMAKE_BINARY_DIR}/external/tmp_install/include")
INCLUDE_DIRECTORIES("${CMAKE_BINARY_DIR}/external/tmp_install/include/TH")

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc")

LINK_DIRECTORIES("${CMAKE_BINARY_DIR}/external/tmp_install/lib") 

list(APPEND mcv_torch_LINKER_LIBS ATen shm  protobuf-nanopb  ${PYTHON_LIBRARY})

list(APPEND torch_files 
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/PtrWrapper.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/Module.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/Generator.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/Size.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/Dtype.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/Exceptions.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/Storage.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/DynamicTypes.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/assertions.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/byte_order.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/utils.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/expand_utils.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/utils/invalid_arguments.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/utils/object_ptr.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/utils/python_arg_parser.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/utils/tensor_list.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/utils/tensor_new.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/utils/tensor_numpy.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/utils/tensor_dtypes.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/utils/tensor_types.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/utils/tuple_parser.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/utils/tensor_apply.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/utils/tensor_flatten.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/utils/variadic.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/allocators.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/serialization.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/init.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/interpreter.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/ir.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/fusion_compiler.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/graph_executor.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/python_ir.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/test_jit.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/tracer.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/tracer_state.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/python_tracer.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/passes/shape_analysis.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/interned_strings.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/type.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/export.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/autodiff.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/interpreter_autograd_function.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/python_arg_flatten.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/python_compiled_function.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/variable_flags.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/passes/create_autodiff_subgraphs.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/passes/graph_fuser.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/passes/onnx.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/passes/dead_code_elimination.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/passes/common_subexpression_elimination.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/passes/peephole.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/passes/inplace_check.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/passes/canonicalize.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/passes/batch_mm.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/passes/onnx/peephole.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/generated/aten_dispatch.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/script/lexer.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/script/compiler.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/script/init.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/jit/script/python_tree_views.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/init.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/grad_mode.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/engine.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/function.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/variable.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/saved_variable.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/input_buffer.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/profiler.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/python_function.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/python_cpp_function.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/python_variable.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/python_variable_indexing.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/python_engine.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/python_hook.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/generated/VariableType.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/generated/Functions.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/generated/python_torch_functions.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/generated/python_variable_methods.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/generated/python_functions.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/generated/python_nn_functions.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/functions/basic_ops.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/functions/tensor.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/functions/accumulate_grad.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/functions/special.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/functions/utils.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/autograd/functions/init.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/tensor/python_tensor.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/onnx/onnx.pb.cpp  
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/onnx/onnx.cpp 
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/generated/TensorDouble.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/generated/TensorFloat.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/generated/TensorHalf.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/generated/TensorLong.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/generated/TensorInt.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/generated/TensorShort.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/generated/TensorChar.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/generated/TensorByte.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/torch/csrc/generated/AutoGPU_cpu_win.cpp
)

list(APPEND source_files ${torch_files})

add_library(mcv_torch MODULE ${source_files})
set_target_properties(mcv_torch  PROPERTIES PREFIX "" OUTPUT_NAME "_C")
set_target_properties(mcv_torch  PROPERTIES SUFFIX ".pyd")
target_link_libraries(mcv_torch ${mcv_torch_LINKER_LIBS})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/torch/pytorch/ DESTINATION python/torch)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/torchvision/ DESTINATION python/torchvision)

install(FILES ${CMAKE_BINARY_DIR}/_C.pyd DESTINATION python/torch)



