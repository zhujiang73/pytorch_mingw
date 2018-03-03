
#include <Python.h>
#include <exception>

#ifdef _THP_CORE
#undef _THP_CORE
#endif

#include "THP_API.h"
#include "torch/csrc/nn/type_checks.h"

#include <TH/TH.h>


TH_API void THNN_FloatAbs_updateOutput(void*, THFloatTensor*, THFloatTensor*);

PyObject * FloatAbs_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatAbs_updateOutput(arg_state, arg_input, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatAbs_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleAbs_updateOutput(void*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleAbs_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleAbs_updateOutput(arg_state, arg_input, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleAbs_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatAbs_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatAbs_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatAbs_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatAbs_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleAbs_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleAbs_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleAbs_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleAbs_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatAbsCriterion_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, bool);

PyObject * FloatAbsCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatAbsCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatAbsCriterion_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor output, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleAbsCriterion_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, bool);

PyObject * DoubleAbsCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleAbsCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleAbsCriterion_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor output, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatAbsCriterion_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, bool);

PyObject * FloatAbsCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatAbsCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatAbsCriterion_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleAbsCriterion_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, bool);

PyObject * DoubleAbsCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleAbsCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleAbsCriterion_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatBCECriterion_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, THFloatTensor*, bool);

PyObject * FloatBCECriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) || PyTuple_GET_ITEM(args, 5) == Py_None) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      THFloatTensor* arg_weights = (PyTuple_GET_ITEM(args, 5) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5)));
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatBCECriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_weights, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatBCECriterion_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor output, bool sizeAverage, [torch.FloatTensor weights or None], bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleBCECriterion_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, THDoubleTensor*, bool);

PyObject * DoubleBCECriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) || PyTuple_GET_ITEM(args, 5) == Py_None) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      THDoubleTensor* arg_weights = (PyTuple_GET_ITEM(args, 5) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5)));
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleBCECriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_weights, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleBCECriterion_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor output, bool sizeAverage, [torch.DoubleTensor weights or None], bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatBCECriterion_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, THFloatTensor*, bool);

PyObject * FloatBCECriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      THFloatTensor* arg_weights = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      bool arg_reduce = (PyTuple_GET_ITEM(args, 7) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatBCECriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_weights, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatBCECriterion_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, bool sizeAverage, [torch.FloatTensor weights or None], bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleBCECriterion_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, THDoubleTensor*, bool);

PyObject * DoubleBCECriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      THDoubleTensor* arg_weights = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      bool arg_reduce = (PyTuple_GET_ITEM(args, 7) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleBCECriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_weights, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleBCECriterion_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, bool sizeAverage, [torch.DoubleTensor weights or None], bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatClassNLLCriterion_updateOutput(void*, THFloatTensor*, THLongTensor*, THFloatTensor*, bool, THFloatTensor*, THFloatTensor*, int64_t, bool);

PyObject * FloatClassNLLCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) || PyTuple_GET_ITEM(args, 5) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      THFloatTensor* arg_weights = (PyTuple_GET_ITEM(args, 5) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5)));
      THFloatTensor* arg_total_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int64_t arg_ignore_index = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      bool arg_reduce = (PyTuple_GET_ITEM(args, 8) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatClassNLLCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_weights, arg_total_weight, arg_ignore_index, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatClassNLLCriterion_updateOutput", 1, "(int state, torch.FloatTensor input, torch.LongTensor target, torch.FloatTensor output, bool sizeAverage, [torch.FloatTensor weights or None], torch.FloatTensor total_weight, int ignore_index, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleClassNLLCriterion_updateOutput(void*, THDoubleTensor*, THLongTensor*, THDoubleTensor*, bool, THDoubleTensor*, THDoubleTensor*, int64_t, bool);

PyObject * DoubleClassNLLCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) || PyTuple_GET_ITEM(args, 5) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      THDoubleTensor* arg_weights = (PyTuple_GET_ITEM(args, 5) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5)));
      THDoubleTensor* arg_total_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int64_t arg_ignore_index = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      bool arg_reduce = (PyTuple_GET_ITEM(args, 8) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleClassNLLCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_weights, arg_total_weight, arg_ignore_index, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleClassNLLCriterion_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.LongTensor target, torch.DoubleTensor output, bool sizeAverage, [torch.DoubleTensor weights or None], torch.DoubleTensor total_weight, int ignore_index, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatClassNLLCriterion_updateGradInput(void*, THFloatTensor*, THLongTensor*, THFloatTensor*, THFloatTensor*, bool, THFloatTensor*, THFloatTensor*, int64_t, bool);

PyObject * FloatClassNLLCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      THFloatTensor* arg_weights = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      THFloatTensor* arg_total_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      int64_t arg_ignore_index = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      bool arg_reduce = (PyTuple_GET_ITEM(args, 9) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatClassNLLCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_weights, arg_total_weight, arg_ignore_index, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatClassNLLCriterion_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.LongTensor target, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, bool sizeAverage, [torch.FloatTensor weights or None], torch.FloatTensor total_weight, int ignore_index, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleClassNLLCriterion_updateGradInput(void*, THDoubleTensor*, THLongTensor*, THDoubleTensor*, THDoubleTensor*, bool, THDoubleTensor*, THDoubleTensor*, int64_t, bool);

PyObject * DoubleClassNLLCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      THDoubleTensor* arg_weights = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      THDoubleTensor* arg_total_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      int64_t arg_ignore_index = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      bool arg_reduce = (PyTuple_GET_ITEM(args, 9) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleClassNLLCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_weights, arg_total_weight, arg_ignore_index, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleClassNLLCriterion_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.LongTensor target, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, bool sizeAverage, [torch.DoubleTensor weights or None], torch.DoubleTensor total_weight, int ignore_index, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialClassNLLCriterion_updateOutput(void*, THFloatTensor*, THLongTensor*, THFloatTensor*, bool, THFloatTensor*, THFloatTensor*, int64_t, bool);

PyObject * FloatSpatialClassNLLCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) || PyTuple_GET_ITEM(args, 5) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      THFloatTensor* arg_weights = (PyTuple_GET_ITEM(args, 5) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5)));
      THFloatTensor* arg_total_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int64_t arg_ignore_index = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      bool arg_reduce = (PyTuple_GET_ITEM(args, 8) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialClassNLLCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_weights, arg_total_weight, arg_ignore_index, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialClassNLLCriterion_updateOutput", 1, "(int state, torch.FloatTensor input, torch.LongTensor target, torch.FloatTensor output, bool sizeAverage, [torch.FloatTensor weights or None], torch.FloatTensor total_weight, int ignore_index, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialClassNLLCriterion_updateOutput(void*, THDoubleTensor*, THLongTensor*, THDoubleTensor*, bool, THDoubleTensor*, THDoubleTensor*, int64_t, bool);

PyObject * DoubleSpatialClassNLLCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) || PyTuple_GET_ITEM(args, 5) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      THDoubleTensor* arg_weights = (PyTuple_GET_ITEM(args, 5) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5)));
      THDoubleTensor* arg_total_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int64_t arg_ignore_index = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      bool arg_reduce = (PyTuple_GET_ITEM(args, 8) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialClassNLLCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_weights, arg_total_weight, arg_ignore_index, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialClassNLLCriterion_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.LongTensor target, torch.DoubleTensor output, bool sizeAverage, [torch.DoubleTensor weights or None], torch.DoubleTensor total_weight, int ignore_index, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialClassNLLCriterion_updateGradInput(void*, THFloatTensor*, THLongTensor*, THFloatTensor*, THFloatTensor*, bool, THFloatTensor*, THFloatTensor*, int64_t, bool);

PyObject * FloatSpatialClassNLLCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      THFloatTensor* arg_weights = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      THFloatTensor* arg_total_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      int64_t arg_ignore_index = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      bool arg_reduce = (PyTuple_GET_ITEM(args, 9) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialClassNLLCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_weights, arg_total_weight, arg_ignore_index, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialClassNLLCriterion_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.LongTensor target, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, bool sizeAverage, [torch.FloatTensor weights or None], torch.FloatTensor total_weight, int ignore_index, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialClassNLLCriterion_updateGradInput(void*, THDoubleTensor*, THLongTensor*, THDoubleTensor*, THDoubleTensor*, bool, THDoubleTensor*, THDoubleTensor*, int64_t, bool);

PyObject * DoubleSpatialClassNLLCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      THDoubleTensor* arg_weights = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      THDoubleTensor* arg_total_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      int64_t arg_ignore_index = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      bool arg_reduce = (PyTuple_GET_ITEM(args, 9) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialClassNLLCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_weights, arg_total_weight, arg_ignore_index, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialClassNLLCriterion_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.LongTensor target, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, bool sizeAverage, [torch.DoubleTensor weights or None], torch.DoubleTensor total_weight, int ignore_index, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatELU_updateOutput(void*, THFloatTensor*, THFloatTensor*, double, double, bool);

PyObject * FloatELU_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_alpha = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatELU_updateOutput(arg_state, arg_input, arg_output, arg_alpha, arg_scale, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatELU_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, float alpha, float scale, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleELU_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, double, double, bool);

PyObject * DoubleELU_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_alpha = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleELU_updateOutput(arg_state, arg_input, arg_output, arg_alpha, arg_scale, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleELU_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, float alpha, float scale, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatELU_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double, double);

PyObject * FloatELU_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_alpha = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatELU_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_output, arg_alpha, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatELU_updateGradInput", 1, "(int state, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor output, float alpha, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleELU_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double, double);

PyObject * DoubleELU_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_alpha = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleELU_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_output, arg_alpha, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleELU_updateGradInput", 1, "(int state, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor output, float alpha, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatDistKLDivCriterion_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, bool);

PyObject * FloatDistKLDivCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatDistKLDivCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatDistKLDivCriterion_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor output, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleDistKLDivCriterion_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, bool);

PyObject * DoubleDistKLDivCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleDistKLDivCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleDistKLDivCriterion_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor output, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatDistKLDivCriterion_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, bool);

PyObject * FloatDistKLDivCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatDistKLDivCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatDistKLDivCriterion_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleDistKLDivCriterion_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, bool);

PyObject * DoubleDistKLDivCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleDistKLDivCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleDistKLDivCriterion_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatGatedLinear_updateOutput(void*, THFloatTensor*, THFloatTensor*, int);

PyObject * FloatGatedLinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_dim = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatGatedLinear_updateOutput(arg_state, arg_input, arg_output, arg_dim);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatGatedLinear_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int dim)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleGatedLinear_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int);

PyObject * DoubleGatedLinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_dim = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleGatedLinear_updateOutput(arg_state, arg_input, arg_output, arg_dim);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleGatedLinear_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int dim)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatGatedLinear_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int);

PyObject * FloatGatedLinear_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_dim = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatGatedLinear_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_dim);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatGatedLinear_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int dim)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleGatedLinear_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int);

PyObject * DoubleGatedLinear_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_dim = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleGatedLinear_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_dim);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleGatedLinear_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int dim)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatHardShrink_updateOutput(void*, THFloatTensor*, THFloatTensor*, double);

PyObject * FloatHardShrink_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_lambda = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatHardShrink_updateOutput(arg_state, arg_input, arg_output, arg_lambda);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatHardShrink_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, float lambda)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleHardShrink_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, double);

PyObject * DoubleHardShrink_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_lambda = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleHardShrink_updateOutput(arg_state, arg_input, arg_output, arg_lambda);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleHardShrink_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, float lambda)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatHardShrink_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double);

PyObject * FloatHardShrink_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_lambda = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatHardShrink_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_lambda);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatHardShrink_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, float lambda)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleHardShrink_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double);

PyObject * DoubleHardShrink_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_lambda = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleHardShrink_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_lambda);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleHardShrink_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, float lambda)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatHardTanh_updateOutput(void*, THFloatTensor*, THFloatTensor*, double, double, bool);

PyObject * FloatHardTanh_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_min_val = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      double arg_max_val = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatHardTanh_updateOutput(arg_state, arg_input, arg_output, arg_min_val, arg_max_val, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatHardTanh_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, float min_val, float max_val, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleHardTanh_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, double, double, bool);

PyObject * DoubleHardTanh_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_min_val = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      double arg_max_val = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleHardTanh_updateOutput(arg_state, arg_input, arg_output, arg_min_val, arg_max_val, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleHardTanh_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, float min_val, float max_val, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatHardTanh_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double, double, bool);

PyObject * FloatHardTanh_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_min_val = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      double arg_max_val = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatHardTanh_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_min_val, arg_max_val, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatHardTanh_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, float min_val, float max_val, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleHardTanh_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double, double, bool);

PyObject * DoubleHardTanh_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_min_val = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      double arg_max_val = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleHardTanh_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_min_val, arg_max_val, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleHardTanh_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, float min_val, float max_val, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatIm2Col_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int);

PyObject * FloatIm2Col_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_sH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_sW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatIm2Col_updateOutput(arg_state, arg_input, arg_output, arg_kH, arg_kW, arg_dH, arg_dW, arg_padH, arg_padW, arg_sH, arg_sW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatIm2Col_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int kH, int kW, int dH, int dW, int padH, int padW, int sH, int sW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleIm2Col_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int);

PyObject * DoubleIm2Col_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_sH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_sW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleIm2Col_updateOutput(arg_state, arg_input, arg_output, arg_kH, arg_kW, arg_dH, arg_dW, arg_padH, arg_padW, arg_sH, arg_sW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleIm2Col_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int kH, int kW, int dH, int dW, int padH, int padW, int sH, int sW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatIm2Col_updateGradInput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int);

PyObject * FloatIm2Col_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_inputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_inputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_sH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_sW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatIm2Col_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_inputHeight, arg_inputWidth, arg_kH, arg_kW, arg_dH, arg_dW, arg_padH, arg_padW, arg_sH, arg_sW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatIm2Col_updateGradInput", 1, "(int state, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int inputHeight, int inputWidth, int kH, int kW, int dH, int dW, int padH, int padW, int sH, int sW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleIm2Col_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int);

PyObject * DoubleIm2Col_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_inputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_inputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_sH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_sW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleIm2Col_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_inputHeight, arg_inputWidth, arg_kH, arg_kW, arg_dH, arg_dW, arg_padH, arg_padW, arg_sH, arg_sW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleIm2Col_updateGradInput", 1, "(int state, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int inputHeight, int inputWidth, int kH, int kW, int dH, int dW, int padH, int padW, int sH, int sW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatCol2Im_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int);

PyObject * FloatCol2Im_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_outputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_outputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_sH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_sW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatCol2Im_updateOutput(arg_state, arg_input, arg_output, arg_outputHeight, arg_outputWidth, arg_kH, arg_kW, arg_dH, arg_dW, arg_padH, arg_padW, arg_sH, arg_sW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatCol2Im_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int outputHeight, int outputWidth, int kH, int kW, int dH, int dW, int padH, int padW, int sH, int sW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleCol2Im_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int);

PyObject * DoubleCol2Im_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_outputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_outputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_sH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_sW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleCol2Im_updateOutput(arg_state, arg_input, arg_output, arg_outputHeight, arg_outputWidth, arg_kH, arg_kW, arg_dH, arg_dW, arg_padH, arg_padW, arg_sH, arg_sW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleCol2Im_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int outputHeight, int outputWidth, int kH, int kW, int dH, int dW, int padH, int padW, int sH, int sW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatCol2Im_updateGradInput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int);

PyObject * FloatCol2Im_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_sH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_sW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatCol2Im_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_kH, arg_kW, arg_dH, arg_dW, arg_padH, arg_padW, arg_sH, arg_sW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatCol2Im_updateGradInput", 1, "(int state, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int kH, int kW, int dH, int dW, int padH, int padW, int sH, int sW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleCol2Im_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int);

PyObject * DoubleCol2Im_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_sH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_sW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleCol2Im_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_kH, arg_kW, arg_dH, arg_dW, arg_padH, arg_padW, arg_sH, arg_sW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleCol2Im_updateGradInput", 1, "(int state, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int kH, int kW, int dH, int dW, int padH, int padW, int sH, int sW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatL1Cost_updateOutput(void*, THFloatTensor*, THFloatTensor*);

PyObject * FloatL1Cost_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatL1Cost_updateOutput(arg_state, arg_input, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatL1Cost_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleL1Cost_updateOutput(void*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleL1Cost_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleL1Cost_updateOutput(arg_state, arg_input, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleL1Cost_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatL1Cost_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatL1Cost_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) || PyTuple_GET_ITEM(args, 2) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = (PyTuple_GET_ITEM(args, 2) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2)));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatL1Cost_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatL1Cost_updateGradInput", 1, "(int state, torch.FloatTensor input, [torch.FloatTensor gradOutput or None], torch.FloatTensor gradInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleL1Cost_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleL1Cost_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) || PyTuple_GET_ITEM(args, 2) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = (PyTuple_GET_ITEM(args, 2) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2)));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleL1Cost_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleL1Cost_updateGradInput", 1, "(int state, torch.DoubleTensor input, [torch.DoubleTensor gradOutput or None], torch.DoubleTensor gradInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatLeakyReLU_updateOutput(void*, THFloatTensor*, THFloatTensor*, double, bool);

PyObject * FloatLeakyReLU_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_negval = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatLeakyReLU_updateOutput(arg_state, arg_input, arg_output, arg_negval, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatLeakyReLU_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, float negval, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleLeakyReLU_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, double, bool);

PyObject * DoubleLeakyReLU_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_negval = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleLeakyReLU_updateOutput(arg_state, arg_input, arg_output, arg_negval, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleLeakyReLU_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, float negval, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatLeakyReLU_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double, bool);

PyObject * FloatLeakyReLU_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_negval = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatLeakyReLU_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_negval, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatLeakyReLU_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, float negval, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleLeakyReLU_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double, bool);

PyObject * DoubleLeakyReLU_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_negval = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleLeakyReLU_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_negval, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleLeakyReLU_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, float negval, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatGRUFused_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatGRUFused_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) || PyTuple_GET_ITEM(args, 3) == Py_None) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_hidden = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_bias1 = (PyTuple_GET_ITEM(args, 3) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3)));
      THFloatTensor* arg_bias2 = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_hx = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THFloatTensor* arg_storage = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatGRUFused_updateOutput(arg_state, arg_input, arg_hidden, arg_bias1, arg_bias2, arg_hx, arg_output, arg_storage);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatGRUFused_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor hidden, [torch.FloatTensor bias1 or None], [torch.FloatTensor bias2 or None], torch.FloatTensor hx, torch.FloatTensor output, torch.FloatTensor storage)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleGRUFused_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleGRUFused_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) || PyTuple_GET_ITEM(args, 3) == Py_None) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_hidden = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_bias1 = (PyTuple_GET_ITEM(args, 3) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3)));
      THDoubleTensor* arg_bias2 = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_hx = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THDoubleTensor* arg_storage = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleGRUFused_updateOutput(arg_state, arg_input, arg_hidden, arg_bias1, arg_bias2, arg_hx, arg_output, arg_storage);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleGRUFused_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor hidden, [torch.DoubleTensor bias1 or None], [torch.DoubleTensor bias2 or None], torch.DoubleTensor hx, torch.DoubleTensor output, torch.DoubleTensor storage)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatGRUFused_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatGRUFused_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_gradInInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradInHidden = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradInputHx = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_storage = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatGRUFused_updateGradInput(arg_state, arg_gradInInput, arg_gradInHidden, arg_gradOutput, arg_gradInputHx, arg_storage);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatGRUFused_updateGradInput", 1, "(int state, torch.FloatTensor gradInInput, torch.FloatTensor gradInHidden, torch.FloatTensor gradOutput, torch.FloatTensor gradInputHx, torch.FloatTensor storage)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleGRUFused_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleGRUFused_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_gradInInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradInHidden = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradInputHx = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_storage = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleGRUFused_updateGradInput(arg_state, arg_gradInInput, arg_gradInHidden, arg_gradOutput, arg_gradInputHx, arg_storage);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleGRUFused_updateGradInput", 1, "(int state, torch.DoubleTensor gradInInput, torch.DoubleTensor gradInHidden, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInputHx, torch.DoubleTensor storage)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatLSTMFused_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatLSTMFused_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) || PyTuple_GET_ITEM(args, 3) == Py_None) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_hidden = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_bias1 = (PyTuple_GET_ITEM(args, 3) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3)));
      THFloatTensor* arg_bias2 = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_cell = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THFloatTensor* arg_outputCell = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatLSTMFused_updateOutput(arg_state, arg_input, arg_hidden, arg_bias1, arg_bias2, arg_cell, arg_output, arg_outputCell);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatLSTMFused_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor hidden, [torch.FloatTensor bias1 or None], [torch.FloatTensor bias2 or None], torch.FloatTensor cell, torch.FloatTensor output, torch.FloatTensor outputCell)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleLSTMFused_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleLSTMFused_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) || PyTuple_GET_ITEM(args, 3) == Py_None) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_hidden = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_bias1 = (PyTuple_GET_ITEM(args, 3) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3)));
      THDoubleTensor* arg_bias2 = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_cell = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THDoubleTensor* arg_outputCell = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleLSTMFused_updateOutput(arg_state, arg_input, arg_hidden, arg_bias1, arg_bias2, arg_cell, arg_output, arg_outputCell);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleLSTMFused_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor hidden, [torch.DoubleTensor bias1 or None], [torch.DoubleTensor bias2 or None], torch.DoubleTensor cell, torch.DoubleTensor output, torch.DoubleTensor outputCell)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatLSTMFused_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatLSTMFused_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_storage = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradInGates = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_cx = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_cy = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_gradOutputCell = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THFloatTensor* arg_gradInputCx = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatLSTMFused_updateGradInput(arg_state, arg_storage, arg_gradInGates, arg_cx, arg_cy, arg_gradOutput, arg_gradOutputCell, arg_gradInputCx);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatLSTMFused_updateGradInput", 1, "(int state, torch.FloatTensor storage, torch.FloatTensor gradInGates, torch.FloatTensor cx, torch.FloatTensor cy, torch.FloatTensor gradOutput, torch.FloatTensor gradOutputCell, torch.FloatTensor gradInputCx)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleLSTMFused_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleLSTMFused_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_storage = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradInGates = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_cx = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_cy = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_gradOutputCell = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THDoubleTensor* arg_gradInputCx = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleLSTMFused_updateGradInput(arg_state, arg_storage, arg_gradInGates, arg_cx, arg_cy, arg_gradOutput, arg_gradOutputCell, arg_gradInputCx);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleLSTMFused_updateGradInput", 1, "(int state, torch.DoubleTensor storage, torch.DoubleTensor gradInGates, torch.DoubleTensor cx, torch.DoubleTensor cy, torch.DoubleTensor gradOutput, torch.DoubleTensor gradOutputCell, torch.DoubleTensor gradInputCx)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatLogSigmoid_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatLogSigmoid_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_buffer = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatLogSigmoid_updateOutput(arg_state, arg_input, arg_output, arg_buffer);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatLogSigmoid_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor buffer)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleLogSigmoid_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleLogSigmoid_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_buffer = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleLogSigmoid_updateOutput(arg_state, arg_input, arg_output, arg_buffer);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleLogSigmoid_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor buffer)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatLogSigmoid_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatLogSigmoid_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_buffer = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatLogSigmoid_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_buffer);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatLogSigmoid_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor buffer)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleLogSigmoid_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleLogSigmoid_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_buffer = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleLogSigmoid_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_buffer);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleLogSigmoid_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor buffer)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatLogSoftMax_updateOutput(void*, THFloatTensor*, THFloatTensor*, int64_t);

PyObject * FloatLogSoftMax_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int64_t arg_dim = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatLogSoftMax_updateOutput(arg_state, arg_input, arg_output, arg_dim);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatLogSoftMax_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int dim)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleLogSoftMax_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int64_t);

PyObject * DoubleLogSoftMax_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int64_t arg_dim = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleLogSoftMax_updateOutput(arg_state, arg_input, arg_output, arg_dim);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleLogSoftMax_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int dim)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatLogSoftMax_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int64_t);

PyObject * FloatLogSoftMax_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int64_t arg_dim = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatLogSoftMax_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_output, arg_dim);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatLogSoftMax_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor output, int dim)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleLogSoftMax_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int64_t);

PyObject * DoubleLogSoftMax_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int64_t arg_dim = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleLogSoftMax_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_output, arg_dim);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleLogSoftMax_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor output, int dim)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatLookupTable_accGradParameters(void*, THLongTensor*, THFloatTensor*, THFloatTensor*, THIntTensor*, THFloatTensor*, THLongTensor*, bool, int, double);

PyObject * FloatLookupTable_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_IntTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) || PyTuple_GET_ITEM(args, 5) == Py_None) &&
          (THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THLongTensor* arg_input = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THIntTensor* arg_count = THNN_IntTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_sorted = (PyTuple_GET_ITEM(args, 5) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5)));
      THLongTensor* arg_indices = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      bool arg_scaleGradByFreq = (PyTuple_GET_ITEM(args, 7) == Py_True ? true : false);
      int arg_paddingValue = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatLookupTable_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_count, arg_sorted, arg_indices, arg_scaleGradByFreq, arg_paddingValue, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatLookupTable_accGradParameters", 1, "(int state, torch.LongTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, torch.IntTensor count, [torch.FloatTensor sorted or None], [torch.LongTensor indices or None], bool scaleGradByFreq, int paddingValue, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleLookupTable_accGradParameters(void*, THLongTensor*, THDoubleTensor*, THDoubleTensor*, THIntTensor*, THDoubleTensor*, THLongTensor*, bool, int, double);

PyObject * DoubleLookupTable_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_IntTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) || PyTuple_GET_ITEM(args, 5) == Py_None) &&
          (THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THLongTensor* arg_input = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THIntTensor* arg_count = THNN_IntTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_sorted = (PyTuple_GET_ITEM(args, 5) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5)));
      THLongTensor* arg_indices = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      bool arg_scaleGradByFreq = (PyTuple_GET_ITEM(args, 7) == Py_True ? true : false);
      int arg_paddingValue = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleLookupTable_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_count, arg_sorted, arg_indices, arg_scaleGradByFreq, arg_paddingValue, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleLookupTable_accGradParameters", 1, "(int state, torch.LongTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, torch.IntTensor count, [torch.DoubleTensor sorted or None], [torch.LongTensor indices or None], bool scaleGradByFreq, int paddingValue, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatLookupTable_renorm(void*, THLongTensor*, THFloatTensor*, double, double);

PyObject * FloatLookupTable_renorm(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THLongTensor* arg_idx = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_maxNorm = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      double arg_normType = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatLookupTable_renorm(arg_state, arg_idx, arg_weight, arg_maxNorm, arg_normType);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatLookupTable_renorm", 1, "(int state, torch.LongTensor idx, torch.FloatTensor weight, float maxNorm, float normType)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleLookupTable_renorm(void*, THLongTensor*, THDoubleTensor*, double, double);

PyObject * DoubleLookupTable_renorm(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THLongTensor* arg_idx = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_maxNorm = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      double arg_normType = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleLookupTable_renorm(arg_state, arg_idx, arg_weight, arg_maxNorm, arg_normType);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleLookupTable_renorm", 1, "(int state, torch.LongTensor idx, torch.DoubleTensor weight, float maxNorm, float normType)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatMarginCriterion_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, double);

PyObject * FloatMarginCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      double arg_margin = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatMarginCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_margin);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatMarginCriterion_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor output, bool sizeAverage, float margin)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleMarginCriterion_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, double);

PyObject * DoubleMarginCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      double arg_margin = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleMarginCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_margin);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleMarginCriterion_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor output, bool sizeAverage, float margin)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatMarginCriterion_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, double);

PyObject * FloatMarginCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      double arg_margin = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatMarginCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradInput, arg_sizeAverage, arg_margin);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatMarginCriterion_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor gradInput, bool sizeAverage, float margin)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleMarginCriterion_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, double);

PyObject * DoubleMarginCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      double arg_margin = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleMarginCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradInput, arg_sizeAverage, arg_margin);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleMarginCriterion_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor gradInput, bool sizeAverage, float margin)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSoftMarginCriterion_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, bool);

PyObject * FloatSoftMarginCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSoftMarginCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSoftMarginCriterion_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor output, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSoftMarginCriterion_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, bool);

PyObject * DoubleSoftMarginCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSoftMarginCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSoftMarginCriterion_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor output, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSoftMarginCriterion_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, bool);

PyObject * FloatSoftMarginCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSoftMarginCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSoftMarginCriterion_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSoftMarginCriterion_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, bool);

PyObject * DoubleSoftMarginCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSoftMarginCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSoftMarginCriterion_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatMSECriterion_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, bool);

PyObject * FloatMSECriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatMSECriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatMSECriterion_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor output, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleMSECriterion_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, bool);

PyObject * DoubleMSECriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleMSECriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleMSECriterion_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor output, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatMSECriterion_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, bool);

PyObject * FloatMSECriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatMSECriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatMSECriterion_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleMSECriterion_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, bool);

PyObject * DoubleMSECriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleMSECriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleMSECriterion_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatMultiLabelMarginCriterion_updateOutput(void*, THFloatTensor*, THLongTensor*, THFloatTensor*, THFloatTensor*, bool, bool);

PyObject * FloatMultiLabelMarginCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_isTarget = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatMultiLabelMarginCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_isTarget, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatMultiLabelMarginCriterion_updateOutput", 1, "(int state, torch.FloatTensor input, torch.LongTensor target, torch.FloatTensor output, torch.FloatTensor isTarget, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleMultiLabelMarginCriterion_updateOutput(void*, THDoubleTensor*, THLongTensor*, THDoubleTensor*, THDoubleTensor*, bool, bool);

PyObject * DoubleMultiLabelMarginCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_isTarget = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleMultiLabelMarginCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_isTarget, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleMultiLabelMarginCriterion_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.LongTensor target, torch.DoubleTensor output, torch.DoubleTensor isTarget, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatMultiLabelMarginCriterion_updateGradInput(void*, THFloatTensor*, THLongTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, bool);

PyObject * FloatMultiLabelMarginCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_isTarget = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 7) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatMultiLabelMarginCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_isTarget, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatMultiLabelMarginCriterion_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.LongTensor target, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor isTarget, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleMultiLabelMarginCriterion_updateGradInput(void*, THDoubleTensor*, THLongTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, bool);

PyObject * DoubleMultiLabelMarginCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_isTarget = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 7) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleMultiLabelMarginCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_isTarget, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleMultiLabelMarginCriterion_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.LongTensor target, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor isTarget, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatMultiMarginCriterion_updateOutput(void*, THFloatTensor*, THLongTensor*, THFloatTensor*, bool, int, THFloatTensor*, double);

PyObject * FloatMultiMarginCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      int arg_p = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_weights = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      double arg_margin = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatMultiMarginCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_p, arg_weights, arg_margin);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatMultiMarginCriterion_updateOutput", 1, "(int state, torch.FloatTensor input, torch.LongTensor target, torch.FloatTensor output, bool sizeAverage, int p, [torch.FloatTensor weights or None], float margin)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleMultiMarginCriterion_updateOutput(void*, THDoubleTensor*, THLongTensor*, THDoubleTensor*, bool, int, THDoubleTensor*, double);

PyObject * DoubleMultiMarginCriterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      int arg_p = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_weights = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      double arg_margin = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleMultiMarginCriterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_p, arg_weights, arg_margin);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleMultiMarginCriterion_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.LongTensor target, torch.DoubleTensor output, bool sizeAverage, int p, [torch.DoubleTensor weights or None], float margin)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatMultiMarginCriterion_updateGradInput(void*, THFloatTensor*, THLongTensor*, THFloatTensor*, bool, int, THFloatTensor*, double);

PyObject * FloatMultiMarginCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      int arg_p = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_weights = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      double arg_margin = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatMultiMarginCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradInput, arg_sizeAverage, arg_p, arg_weights, arg_margin);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatMultiMarginCriterion_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.LongTensor target, torch.FloatTensor gradInput, bool sizeAverage, int p, [torch.FloatTensor weights or None], float margin)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleMultiMarginCriterion_updateGradInput(void*, THDoubleTensor*, THLongTensor*, THDoubleTensor*, bool, int, THDoubleTensor*, double);

PyObject * DoubleMultiMarginCriterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THLongTensor* arg_target = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      int arg_p = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_weights = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      double arg_margin = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleMultiMarginCriterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradInput, arg_sizeAverage, arg_p, arg_weights, arg_margin);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleMultiMarginCriterion_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.LongTensor target, torch.DoubleTensor gradInput, bool sizeAverage, int p, [torch.DoubleTensor weights or None], float margin)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatPReLU_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatPReLU_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatPReLU_updateOutput(arg_state, arg_input, arg_output, arg_weight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatPReLU_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoublePReLU_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoublePReLU_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoublePReLU_updateOutput(arg_state, arg_input, arg_output, arg_weight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoublePReLU_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatPReLU_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatPReLU_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatPReLU_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatPReLU_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoublePReLU_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoublePReLU_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoublePReLU_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoublePReLU_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatPReLU_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double);

PyObject * FloatPReLU_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatPReLU_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_gradWeight, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatPReLU_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor gradWeight, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoublePReLU_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double);

PyObject * DoublePReLU_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoublePReLU_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_gradWeight, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoublePReLU_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor gradWeight, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatLinear_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatLinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_addBuffer = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatLinear_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_addBuffer);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatLinear_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, torch.FloatTensor bias, torch.FloatTensor addBuffer)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleLinear_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleLinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_addBuffer = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleLinear_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_addBuffer);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleLinear_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, torch.DoubleTensor bias, torch.DoubleTensor addBuffer)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatLinear_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatLinear_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatLinear_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatLinear_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleLinear_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleLinear_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleLinear_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleLinear_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatLinear_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double);

PyObject * FloatLinear_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 8)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      THFloatTensor* arg_addBuffer = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatLinear_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_bias, arg_gradWeight, arg_gradBias, arg_addBuffer, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatLinear_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor bias, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, torch.FloatTensor addBuffer, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleLinear_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double);

PyObject * DoubleLinear_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 8)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      THDoubleTensor* arg_addBuffer = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleLinear_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_bias, arg_gradWeight, arg_gradBias, arg_addBuffer, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleLinear_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor bias, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, torch.DoubleTensor addBuffer, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatRReLU_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double, double, bool, bool, THGenerator*);

PyObject * FloatRReLU_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 7)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 8)) == THPGeneratorClass) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_noise = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_lower = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      double arg_upper = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      bool arg_train = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      bool arg_inplace = (PyTuple_GET_ITEM(args, 7) == Py_True ? true : false);
      THGenerator* arg_generator = THPGenerator_TH_CData((THPGenerator*)PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatRReLU_updateOutput(arg_state, arg_input, arg_output, arg_noise, arg_lower, arg_upper, arg_train, arg_inplace, arg_generator);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatRReLU_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor noise, float lower, float upper, bool train, bool inplace, Generator generator)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleRReLU_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double, double, bool, bool, THGenerator*);

PyObject * DoubleRReLU_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 7)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 8)) == THPGeneratorClass) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_noise = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_lower = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      double arg_upper = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      bool arg_train = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      bool arg_inplace = (PyTuple_GET_ITEM(args, 7) == Py_True ? true : false);
      THGenerator* arg_generator = THPGenerator_TH_CData((THPGenerator*)PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleRReLU_updateOutput(arg_state, arg_input, arg_output, arg_noise, arg_lower, arg_upper, arg_train, arg_inplace, arg_generator);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleRReLU_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor noise, float lower, float upper, bool train, bool inplace, Generator generator)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatRReLU_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double, double, bool, bool);

PyObject * FloatRReLU_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 6)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 7)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_noise = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      double arg_lower = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      double arg_upper = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 6));
      bool arg_train = (PyTuple_GET_ITEM(args, 7) == Py_True ? true : false);
      bool arg_inplace = (PyTuple_GET_ITEM(args, 8) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatRReLU_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_noise, arg_lower, arg_upper, arg_train, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatRReLU_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor noise, float lower, float upper, bool train, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleRReLU_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double, double, bool, bool);

PyObject * DoubleRReLU_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 6)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 7)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_noise = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      double arg_lower = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      double arg_upper = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 6));
      bool arg_train = (PyTuple_GET_ITEM(args, 7) == Py_True ? true : false);
      bool arg_inplace = (PyTuple_GET_ITEM(args, 8) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleRReLU_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_noise, arg_lower, arg_upper, arg_train, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleRReLU_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor noise, float lower, float upper, bool train, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSigmoid_updateOutput(void*, THFloatTensor*, THFloatTensor*);

PyObject * FloatSigmoid_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSigmoid_updateOutput(arg_state, arg_input, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSigmoid_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSigmoid_updateOutput(void*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleSigmoid_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSigmoid_updateOutput(arg_state, arg_input, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSigmoid_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSigmoid_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatSigmoid_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSigmoid_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSigmoid_updateGradInput", 1, "(int state, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSigmoid_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleSigmoid_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSigmoid_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSigmoid_updateGradInput", 1, "(int state, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSmoothL1Criterion_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, bool);

PyObject * FloatSmoothL1Criterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSmoothL1Criterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSmoothL1Criterion_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor output, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSmoothL1Criterion_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, bool);

PyObject * DoubleSmoothL1Criterion_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 4) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSmoothL1Criterion_updateOutput(arg_state, arg_input, arg_target, arg_output, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSmoothL1Criterion_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor output, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSmoothL1Criterion_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, bool);

PyObject * FloatSmoothL1Criterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_target = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSmoothL1Criterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSmoothL1Criterion_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor target, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSmoothL1Criterion_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, bool);

PyObject * DoubleSmoothL1Criterion_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_target = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      bool arg_sizeAverage = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      bool arg_reduce = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSmoothL1Criterion_updateGradInput(arg_state, arg_input, arg_target, arg_gradOutput, arg_gradInput, arg_sizeAverage, arg_reduce);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSmoothL1Criterion_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor target, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, bool sizeAverage, bool reduce)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSoftMax_updateOutput(void*, THFloatTensor*, THFloatTensor*, int64_t);

PyObject * FloatSoftMax_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int64_t arg_dim = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSoftMax_updateOutput(arg_state, arg_input, arg_output, arg_dim);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSoftMax_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int dim)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSoftMax_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int64_t);

PyObject * DoubleSoftMax_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int64_t arg_dim = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSoftMax_updateOutput(arg_state, arg_input, arg_output, arg_dim);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSoftMax_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int dim)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSoftMax_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int64_t);

PyObject * FloatSoftMax_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int64_t arg_dim = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSoftMax_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_output, arg_dim);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSoftMax_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor output, int dim)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSoftMax_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int64_t);

PyObject * DoubleSoftMax_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int64_t arg_dim = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSoftMax_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_output, arg_dim);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSoftMax_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor output, int dim)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSoftPlus_updateOutput(void*, THFloatTensor*, THFloatTensor*, double, double);

PyObject * FloatSoftPlus_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_beta = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      double arg_threshold = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSoftPlus_updateOutput(arg_state, arg_input, arg_output, arg_beta, arg_threshold);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSoftPlus_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, float beta, float threshold)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSoftPlus_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, double, double);

PyObject * DoubleSoftPlus_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_beta = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      double arg_threshold = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSoftPlus_updateOutput(arg_state, arg_input, arg_output, arg_beta, arg_threshold);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSoftPlus_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, float beta, float threshold)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSoftPlus_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double, double);

PyObject * FloatSoftPlus_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      double arg_beta = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      double arg_threshold = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSoftPlus_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_output, arg_beta, arg_threshold);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSoftPlus_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor output, float beta, float threshold)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSoftPlus_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double, double);

PyObject * DoubleSoftPlus_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      double arg_beta = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      double arg_threshold = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSoftPlus_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_output, arg_beta, arg_threshold);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSoftPlus_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor output, float beta, float threshold)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSoftShrink_updateOutput(void*, THFloatTensor*, THFloatTensor*, double);

PyObject * FloatSoftShrink_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_lambda = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSoftShrink_updateOutput(arg_state, arg_input, arg_output, arg_lambda);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSoftShrink_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, float lambda)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSoftShrink_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, double);

PyObject * DoubleSoftShrink_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_lambda = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSoftShrink_updateOutput(arg_state, arg_input, arg_output, arg_lambda);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSoftShrink_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, float lambda)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSoftShrink_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double);

PyObject * FloatSoftShrink_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_lambda = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSoftShrink_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_lambda);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSoftShrink_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, float lambda)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSoftShrink_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double);

PyObject * DoubleSoftShrink_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_lambda = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSoftShrink_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_lambda);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSoftShrink_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, float lambda)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatIndexLinear_updateOutput(void*, THLongTensor*, int64_t, THFloatTensor*, THLongTensor*, THLongTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int);

PyObject * FloatIndexLinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 8)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THLongTensor* arg_keys = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      int64_t arg_keysOffset = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_values = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_sizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THLongTensor* arg_cumSumSizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      THFloatTensor* arg_normalizedValues = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 9));
      int arg_train = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatIndexLinear_updateOutput(arg_state, arg_keys, arg_keysOffset, arg_values, arg_sizes, arg_cumSumSizes, arg_output, arg_weight, arg_bias, arg_normalizedValues, arg_train);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatIndexLinear_updateOutput", 1, "(int state, torch.LongTensor keys, int keysOffset, torch.FloatTensor values, torch.LongTensor sizes, torch.LongTensor cumSumSizes, torch.FloatTensor output, torch.FloatTensor weight, torch.FloatTensor bias, torch.FloatTensor normalizedValues, int train)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleIndexLinear_updateOutput(void*, THLongTensor*, int64_t, THDoubleTensor*, THLongTensor*, THLongTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int);

PyObject * DoubleIndexLinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 8)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THLongTensor* arg_keys = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      int64_t arg_keysOffset = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_values = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_sizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THLongTensor* arg_cumSumSizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      THDoubleTensor* arg_normalizedValues = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 9));
      int arg_train = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleIndexLinear_updateOutput(arg_state, arg_keys, arg_keysOffset, arg_values, arg_sizes, arg_cumSumSizes, arg_output, arg_weight, arg_bias, arg_normalizedValues, arg_train);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleIndexLinear_updateOutput", 1, "(int state, torch.LongTensor keys, int keysOffset, torch.DoubleTensor values, torch.LongTensor sizes, torch.LongTensor cumSumSizes, torch.DoubleTensor output, torch.DoubleTensor weight, torch.DoubleTensor bias, torch.DoubleTensor normalizedValues, int train)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatIndexLinear_accGradParameters(void*, THLongTensor*, int64_t, THFloatTensor*, THLongTensor*, THLongTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double, double);

PyObject * FloatIndexLinear_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 8)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 9)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 10)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 11)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 12)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THLongTensor* arg_keys = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      int64_t arg_keysOffset = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_values = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_sizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THLongTensor* arg_cumSumSizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 9));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 10));
      THFloatTensor* arg_valuesBuffer = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 11));
      double arg_weightDecay = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 12));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatIndexLinear_accGradParameters(arg_state, arg_keys, arg_keysOffset, arg_values, arg_sizes, arg_cumSumSizes, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_weight, arg_bias, arg_valuesBuffer, arg_weightDecay, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatIndexLinear_accGradParameters", 1, "(int state, torch.LongTensor keys, int keysOffset, torch.FloatTensor values, torch.LongTensor sizes, torch.LongTensor cumSumSizes, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, torch.FloatTensor weight, torch.FloatTensor bias, torch.FloatTensor valuesBuffer, float weightDecay, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleIndexLinear_accGradParameters(void*, THLongTensor*, int64_t, THDoubleTensor*, THLongTensor*, THLongTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double, double);

PyObject * DoubleIndexLinear_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 8)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 9)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 10)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 11)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 12)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THLongTensor* arg_keys = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      int64_t arg_keysOffset = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_values = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_sizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THLongTensor* arg_cumSumSizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 9));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 10));
      THDoubleTensor* arg_valuesBuffer = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 11));
      double arg_weightDecay = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 12));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleIndexLinear_accGradParameters(arg_state, arg_keys, arg_keysOffset, arg_values, arg_sizes, arg_cumSumSizes, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_weight, arg_bias, arg_valuesBuffer, arg_weightDecay, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleIndexLinear_accGradParameters", 1, "(int state, torch.LongTensor keys, int keysOffset, torch.DoubleTensor values, torch.LongTensor sizes, torch.LongTensor cumSumSizes, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, torch.DoubleTensor weight, torch.DoubleTensor bias, torch.DoubleTensor valuesBuffer, float weightDecay, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatIndexLinear_accUpdateGradParameters(void*, THLongTensor*, int64_t, THFloatTensor*, THLongTensor*, THLongTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double, double);

PyObject * FloatIndexLinear_accUpdateGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 8)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 9)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THLongTensor* arg_keys = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      int64_t arg_keysOffset = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_values = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_sizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THLongTensor* arg_cumSumSizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      double arg_weightDecay = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 9));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatIndexLinear_accUpdateGradParameters(arg_state, arg_keys, arg_keysOffset, arg_values, arg_sizes, arg_cumSumSizes, arg_gradOutput, arg_weight, arg_bias, arg_weightDecay, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatIndexLinear_accUpdateGradParameters", 1, "(int state, torch.LongTensor keys, int keysOffset, torch.FloatTensor values, torch.LongTensor sizes, torch.LongTensor cumSumSizes, torch.FloatTensor gradOutput, torch.FloatTensor weight, torch.FloatTensor bias, float weightDecay, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleIndexLinear_accUpdateGradParameters(void*, THLongTensor*, int64_t, THDoubleTensor*, THLongTensor*, THLongTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double, double);

PyObject * DoubleIndexLinear_accUpdateGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 8)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 9)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THLongTensor* arg_keys = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      int64_t arg_keysOffset = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_values = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_sizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THLongTensor* arg_cumSumSizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      double arg_weightDecay = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 9));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleIndexLinear_accUpdateGradParameters(arg_state, arg_keys, arg_keysOffset, arg_values, arg_sizes, arg_cumSumSizes, arg_gradOutput, arg_weight, arg_bias, arg_weightDecay, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleIndexLinear_accUpdateGradParameters", 1, "(int state, torch.LongTensor keys, int keysOffset, torch.DoubleTensor values, torch.LongTensor sizes, torch.LongTensor cumSumSizes, torch.DoubleTensor gradOutput, torch.DoubleTensor weight, torch.DoubleTensor bias, float weightDecay, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatIndexLinear_updateParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THLongTensor*, THLongTensor*, int64_t, double, double);

PyObject * FloatIndexLinear_updateParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 8)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THLongTensor* arg_runningKeys = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THLongTensor* arg_cumSumSizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int64_t arg_keysOffset = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      double arg_weightDecay = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 8));
      double arg_learningRate = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatIndexLinear_updateParameters(arg_state, arg_gradWeight, arg_gradBias, arg_weight, arg_bias, arg_runningKeys, arg_cumSumSizes, arg_keysOffset, arg_weightDecay, arg_learningRate);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatIndexLinear_updateParameters", 1, "(int state, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, torch.FloatTensor weight, torch.FloatTensor bias, torch.LongTensor runningKeys, torch.LongTensor cumSumSizes, int keysOffset, float weightDecay, float learningRate)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleIndexLinear_updateParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, THLongTensor*, int64_t, double, double);

PyObject * DoubleIndexLinear_updateParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 8)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THLongTensor* arg_runningKeys = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THLongTensor* arg_cumSumSizes = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int64_t arg_keysOffset = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      double arg_weightDecay = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 8));
      double arg_learningRate = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleIndexLinear_updateParameters(arg_state, arg_gradWeight, arg_gradBias, arg_weight, arg_bias, arg_runningKeys, arg_cumSumSizes, arg_keysOffset, arg_weightDecay, arg_learningRate);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleIndexLinear_updateParameters", 1, "(int state, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, torch.DoubleTensor weight, torch.DoubleTensor bias, torch.LongTensor runningKeys, torch.LongTensor cumSumSizes, int keysOffset, float weightDecay, float learningRate)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSparseLinear_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatSparseLinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSparseLinear_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSparseLinear_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, torch.FloatTensor bias)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSparseLinear_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleSparseLinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSparseLinear_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSparseLinear_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, torch.DoubleTensor bias)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSparseLinear_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double, double);

PyObject * FloatSparseLinear_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 7)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      double arg_weightDecay = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 7));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSparseLinear_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_weight, arg_bias, arg_weightDecay, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSparseLinear_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, torch.FloatTensor weight, torch.FloatTensor bias, float weightDecay, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSparseLinear_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double, double);

PyObject * DoubleSparseLinear_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 7)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      double arg_weightDecay = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 7));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSparseLinear_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_weight, arg_bias, arg_weightDecay, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSparseLinear_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, torch.DoubleTensor weight, torch.DoubleTensor bias, float weightDecay, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSparseLinear_zeroGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatSparseLinear_zeroGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_lastInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSparseLinear_zeroGradParameters(arg_state, arg_gradWeight, arg_gradBias, arg_lastInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSparseLinear_zeroGradParameters", 1, "(int state, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, torch.FloatTensor lastInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSparseLinear_zeroGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleSparseLinear_zeroGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_lastInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSparseLinear_zeroGradParameters(arg_state, arg_gradWeight, arg_gradBias, arg_lastInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSparseLinear_zeroGradParameters", 1, "(int state, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, torch.DoubleTensor lastInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSparseLinear_updateParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double);

PyObject * FloatSparseLinear_updateParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_lastInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      double arg_learningRate = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSparseLinear_updateParameters(arg_state, arg_weight, arg_bias, arg_gradWeight, arg_gradBias, arg_lastInput, arg_learningRate);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSparseLinear_updateParameters", 1, "(int state, torch.FloatTensor weight, torch.FloatTensor bias, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, torch.FloatTensor lastInput, float learningRate)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSparseLinear_updateParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double);

PyObject * DoubleSparseLinear_updateParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_lastInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      double arg_learningRate = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSparseLinear_updateParameters(arg_state, arg_weight, arg_bias, arg_gradWeight, arg_gradBias, arg_lastInput, arg_learningRate);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSparseLinear_updateParameters", 1, "(int state, torch.DoubleTensor weight, torch.DoubleTensor bias, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, torch.DoubleTensor lastInput, float learningRate)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSparseLinear_legacyUpdateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatSparseLinear_legacyUpdateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSparseLinear_legacyUpdateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSparseLinear_legacyUpdateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, torch.FloatTensor bias)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSparseLinear_legacyUpdateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleSparseLinear_legacyUpdateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSparseLinear_legacyUpdateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSparseLinear_legacyUpdateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, torch.DoubleTensor bias)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSparseLinear_legacyAccGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double, double);

PyObject * FloatSparseLinear_legacyAccGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 7)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      double arg_weightDecay = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 7));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSparseLinear_legacyAccGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_weight, arg_bias, arg_weightDecay, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSparseLinear_legacyAccGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, torch.FloatTensor weight, torch.FloatTensor bias, float weightDecay, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSparseLinear_legacyAccGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double, double);

PyObject * DoubleSparseLinear_legacyAccGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 7)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      double arg_weightDecay = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 7));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSparseLinear_legacyAccGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_weight, arg_bias, arg_weightDecay, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSparseLinear_legacyAccGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, torch.DoubleTensor weight, torch.DoubleTensor bias, float weightDecay, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSparseLinear_legacyZeroGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatSparseLinear_legacyZeroGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_lastInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSparseLinear_legacyZeroGradParameters(arg_state, arg_gradWeight, arg_gradBias, arg_lastInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSparseLinear_legacyZeroGradParameters", 1, "(int state, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, torch.FloatTensor lastInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSparseLinear_legacyZeroGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleSparseLinear_legacyZeroGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_lastInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSparseLinear_legacyZeroGradParameters(arg_state, arg_gradWeight, arg_gradBias, arg_lastInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSparseLinear_legacyZeroGradParameters", 1, "(int state, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, torch.DoubleTensor lastInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSparseLinear_legacyUpdateParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double);

PyObject * FloatSparseLinear_legacyUpdateParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_lastInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      double arg_learningRate = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSparseLinear_legacyUpdateParameters(arg_state, arg_weight, arg_bias, arg_gradWeight, arg_gradBias, arg_lastInput, arg_learningRate);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSparseLinear_legacyUpdateParameters", 1, "(int state, torch.FloatTensor weight, torch.FloatTensor bias, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, torch.FloatTensor lastInput, float learningRate)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSparseLinear_legacyUpdateParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double);

PyObject * DoubleSparseLinear_legacyUpdateParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_lastInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      double arg_learningRate = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSparseLinear_legacyUpdateParameters(arg_state, arg_weight, arg_bias, arg_gradWeight, arg_gradBias, arg_lastInput, arg_learningRate);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSparseLinear_legacyUpdateParameters", 1, "(int state, torch.DoubleTensor weight, torch.DoubleTensor bias, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, torch.DoubleTensor lastInput, float learningRate)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSqrt_updateOutput(void*, THFloatTensor*, THFloatTensor*, double);

PyObject * FloatSqrt_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_eps = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSqrt_updateOutput(arg_state, arg_input, arg_output, arg_eps);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSqrt_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, float eps)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSqrt_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, double);

PyObject * DoubleSqrt_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_eps = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSqrt_updateOutput(arg_state, arg_input, arg_output, arg_eps);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSqrt_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, float eps)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSqrt_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatSqrt_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSqrt_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSqrt_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSqrt_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleSqrt_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSqrt_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSqrt_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSquare_updateOutput(void*, THFloatTensor*, THFloatTensor*);

PyObject * FloatSquare_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSquare_updateOutput(arg_state, arg_input, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSquare_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSquare_updateOutput(void*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleSquare_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSquare_updateOutput(arg_state, arg_input, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSquare_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSquare_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatSquare_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSquare_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSquare_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSquare_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleSquare_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSquare_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSquare_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTanh_updateOutput(void*, THFloatTensor*, THFloatTensor*);

PyObject * FloatTanh_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTanh_updateOutput(arg_state, arg_input, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTanh_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTanh_updateOutput(void*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleTanh_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTanh_updateOutput(arg_state, arg_input, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTanh_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTanh_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatTanh_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTanh_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTanh_updateGradInput", 1, "(int state, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTanh_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleTanh_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTanh_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_output);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTanh_updateGradInput", 1, "(int state, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor output)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatThreshold_updateOutput(void*, THFloatTensor*, THFloatTensor*, double, double, bool);

PyObject * FloatThreshold_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_threshold = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      double arg_val = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatThreshold_updateOutput(arg_state, arg_input, arg_output, arg_threshold, arg_val, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatThreshold_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, float threshold, float val, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleThreshold_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, double, double, bool);

PyObject * DoubleThreshold_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_threshold = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      double arg_val = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 5) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleThreshold_updateOutput(arg_state, arg_input, arg_output, arg_threshold, arg_val, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleThreshold_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, float threshold, float val, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatThreshold_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double, double, bool);

PyObject * FloatThreshold_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_threshold = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      double arg_val = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatThreshold_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_threshold, arg_val, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatThreshold_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, float threshold, float val, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleThreshold_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double, double, bool);

PyObject * DoubleThreshold_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      double arg_threshold = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 4));
      double arg_val = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      bool arg_inplace = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleThreshold_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_threshold, arg_val, arg_inplace);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleThreshold_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, float threshold, float val, bool inplace)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalConvolution_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int);

PyObject * FloatTemporalConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_inputFrameSize = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_outputFrameSize = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_kW, arg_dW, arg_inputFrameSize, arg_outputFrameSize);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalConvolution_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, torch.FloatTensor bias, int kW, int dW, int inputFrameSize, int outputFrameSize)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalConvolution_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int);

PyObject * DoubleTemporalConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_inputFrameSize = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_outputFrameSize = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_kW, arg_dW, arg_inputFrameSize, arg_outputFrameSize);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalConvolution_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, torch.DoubleTensor bias, int kW, int dW, int inputFrameSize, int outputFrameSize)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalConvolution_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int);

PyObject * FloatTemporalConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_kW, arg_dW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalConvolution_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, int kW, int dW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalConvolution_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int);

PyObject * DoubleTemporalConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_kW, arg_dW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalConvolution_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, int kW, int dW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalConvolution_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, double);

PyObject * FloatTemporalConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_kW, arg_dW, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalConvolution_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, int kW, int dW, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalConvolution_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, double);

PyObject * DoubleTemporalConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_kW, arg_dW, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalConvolution_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, int kW, int dW, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalMaxPooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int);

PyObject * FloatTemporalMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_kW, arg_dW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalMaxPooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.LongTensor indices, int kW, int dW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalMaxPooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int);

PyObject * DoubleTemporalMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_kW, arg_dW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalMaxPooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.LongTensor indices, int kW, int dW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalMaxPooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int);

PyObject * FloatTemporalMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_kW, arg_dW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalMaxPooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.LongTensor indices, int kW, int dW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalMaxPooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int);

PyObject * DoubleTemporalMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_kW, arg_dW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalMaxPooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.LongTensor indices, int kW, int dW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalSubSampling_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int);

PyObject * FloatTemporalSubSampling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_inputFrameSize = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalSubSampling_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_kW, arg_dW, arg_inputFrameSize);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalSubSampling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, torch.FloatTensor bias, int kW, int dW, int inputFrameSize)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalSubSampling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int);

PyObject * DoubleTemporalSubSampling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_inputFrameSize = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalSubSampling_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_kW, arg_dW, arg_inputFrameSize);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalSubSampling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, torch.DoubleTensor bias, int kW, int dW, int inputFrameSize)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalSubSampling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int);

PyObject * FloatTemporalSubSampling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalSubSampling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_kW, arg_dW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalSubSampling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, int kW, int dW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalSubSampling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int);

PyObject * DoubleTemporalSubSampling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalSubSampling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_kW, arg_dW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalSubSampling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, int kW, int dW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalSubSampling_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, double);

PyObject * FloatTemporalSubSampling_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalSubSampling_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_kW, arg_dW, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalSubSampling_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, int kW, int dW, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalSubSampling_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, double);

PyObject * DoubleTemporalSubSampling_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalSubSampling_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_kW, arg_dW, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalSubSampling_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, int kW, int dW, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalRowConvolution_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, bool);

PyObject * FloatTemporalRowConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      bool arg_featFirst = (PyTuple_GET_ITEM(args, 10) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalRowConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_kW, arg_dW, arg_padW, arg_featFirst);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalRowConvolution_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, torch.FloatTensor bias, torch.FloatTensor finput, torch.FloatTensor fgradInput, int kW, int dW, int padW, bool featFirst)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalRowConvolution_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, bool);

PyObject * DoubleTemporalRowConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      bool arg_featFirst = (PyTuple_GET_ITEM(args, 10) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalRowConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_kW, arg_dW, arg_padW, arg_featFirst);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalRowConvolution_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, torch.DoubleTensor bias, torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kW, int dW, int padW, bool featFirst)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalRowConvolution_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, bool);

PyObject * FloatTemporalRowConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      bool arg_featFirst = (PyTuple_GET_ITEM(args, 10) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalRowConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_fgradInput, arg_kW, arg_dW, arg_padW, arg_featFirst);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalRowConvolution_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor finput, torch.FloatTensor fgradInput, int kW, int dW, int padW, bool featFirst)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalRowConvolution_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, bool);

PyObject * DoubleTemporalRowConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      bool arg_featFirst = (PyTuple_GET_ITEM(args, 10) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalRowConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_fgradInput, arg_kW, arg_dW, arg_padW, arg_featFirst);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalRowConvolution_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kW, int dW, int padW, bool featFirst)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalRowConvolution_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, bool, double);

PyObject * FloatTemporalRowConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 12 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 10)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 11))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      bool arg_featFirst = (PyTuple_GET_ITEM(args, 10) == Py_True ? true : false);
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 11));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalRowConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_kW, arg_dW, arg_padW, arg_featFirst, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalRowConvolution_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, torch.FloatTensor finput, torch.FloatTensor fgradInput, int kW, int dW, int padW, bool featFirst, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalRowConvolution_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, bool, double);

PyObject * DoubleTemporalRowConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 12 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 10)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 11))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      bool arg_featFirst = (PyTuple_GET_ITEM(args, 10) == Py_True ? true : false);
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 11));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalRowConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_kW, arg_dW, arg_padW, arg_featFirst, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalRowConvolution_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kW, int dW, int padW, bool featFirst, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalUpSamplingNearest_updateOutput(void*, THFloatTensor*, THFloatTensor*, int);

PyObject * FloatTemporalUpSamplingNearest_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_scale_factor = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalUpSamplingNearest_updateOutput(arg_state, arg_input, arg_output, arg_scale_factor);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalUpSamplingNearest_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int scale_factor)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalUpSamplingNearest_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int);

PyObject * DoubleTemporalUpSamplingNearest_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_scale_factor = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalUpSamplingNearest_updateOutput(arg_state, arg_input, arg_output, arg_scale_factor);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalUpSamplingNearest_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int scale_factor)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalUpSamplingNearest_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int);

PyObject * FloatTemporalUpSamplingNearest_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_scale_factor = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalUpSamplingNearest_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_scale_factor);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalUpSamplingNearest_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int scale_factor)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalUpSamplingNearest_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int);

PyObject * DoubleTemporalUpSamplingNearest_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_scale_factor = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalUpSamplingNearest_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_scale_factor);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalUpSamplingNearest_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int scale_factor)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalUpSamplingLinear_updateOutput(void*, THFloatTensor*, THFloatTensor*, int);

PyObject * FloatTemporalUpSamplingLinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalUpSamplingLinear_updateOutput(arg_state, arg_input, arg_output, arg_osizeW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalUpSamplingLinear_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int osizeW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalUpSamplingLinear_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int);

PyObject * DoubleTemporalUpSamplingLinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalUpSamplingLinear_updateOutput(arg_state, arg_input, arg_output, arg_osizeW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalUpSamplingLinear_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int osizeW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalUpSamplingLinear_updateGradInput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int);

PyObject * FloatTemporalUpSamplingLinear_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_isizeB = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_isizeC = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_isizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalUpSamplingLinear_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_isizeB, arg_isizeC, arg_isizeW, arg_osizeW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalUpSamplingLinear_updateGradInput", 1, "(int state, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int isizeB, int isizeC, int isizeW, int osizeW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalUpSamplingLinear_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int);

PyObject * DoubleTemporalUpSamplingLinear_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_isizeB = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_isizeC = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_isizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalUpSamplingLinear_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_isizeB, arg_isizeC, arg_isizeW, arg_osizeW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalUpSamplingLinear_updateGradInput", 1, "(int state, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int isizeB, int isizeC, int isizeW, int osizeW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatBatchNormalization_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, double, double);

PyObject * FloatBatchNormalization_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 12 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) || PyTuple_GET_ITEM(args, 3) == Py_None) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) || PyTuple_GET_ITEM(args, 5) == Py_None) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 8)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 9)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 10)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 11))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = (PyTuple_GET_ITEM(args, 3) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3)));
      THFloatTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_running_mean = (PyTuple_GET_ITEM(args, 5) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5)));
      THFloatTensor* arg_running_var = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      THFloatTensor* arg_save_mean = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      THFloatTensor* arg_save_std = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      bool arg_train = (PyTuple_GET_ITEM(args, 9) == Py_True ? true : false);
      double arg_momentum = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 10));
      double arg_eps = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 11));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatBatchNormalization_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_running_mean, arg_running_var, arg_save_mean, arg_save_std, arg_train, arg_momentum, arg_eps);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatBatchNormalization_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, [torch.FloatTensor weight or None], [torch.FloatTensor bias or None], [torch.FloatTensor running_mean or None], [torch.FloatTensor running_var or None], torch.FloatTensor save_mean, torch.FloatTensor save_std, bool train, float momentum, float eps)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleBatchNormalization_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, double, double);

PyObject * DoubleBatchNormalization_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 12 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) || PyTuple_GET_ITEM(args, 3) == Py_None) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) || PyTuple_GET_ITEM(args, 5) == Py_None) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 8)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 9)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 10)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 11))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = (PyTuple_GET_ITEM(args, 3) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3)));
      THDoubleTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_running_mean = (PyTuple_GET_ITEM(args, 5) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5)));
      THDoubleTensor* arg_running_var = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      THDoubleTensor* arg_save_mean = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      THDoubleTensor* arg_save_std = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      bool arg_train = (PyTuple_GET_ITEM(args, 9) == Py_True ? true : false);
      double arg_momentum = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 10));
      double arg_eps = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 11));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleBatchNormalization_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_running_mean, arg_running_var, arg_save_mean, arg_save_std, arg_train, arg_momentum, arg_eps);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleBatchNormalization_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, [torch.DoubleTensor weight or None], [torch.DoubleTensor bias or None], [torch.DoubleTensor running_mean or None], [torch.DoubleTensor running_var or None], torch.DoubleTensor save_mean, torch.DoubleTensor save_std, bool train, float momentum, float eps)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatBatchNormalization_backward(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, bool, double, double);

PyObject * FloatBatchNormalization_backward(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) || PyTuple_GET_ITEM(args, 3) == Py_None) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) || PyTuple_GET_ITEM(args, 5) == Py_None) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 7)) || PyTuple_GET_ITEM(args, 7) == Py_None) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 8)) || PyTuple_GET_ITEM(args, 8) == Py_None) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 9)) || PyTuple_GET_ITEM(args, 9) == Py_None) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 10)) || PyTuple_GET_ITEM(args, 10) == Py_None) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 11)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 12)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = (PyTuple_GET_ITEM(args, 3) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3)));
      THFloatTensor* arg_gradWeight = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 5) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5)));
      THFloatTensor* arg_weight = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      THFloatTensor* arg_running_mean = (PyTuple_GET_ITEM(args, 7) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 7)));
      THFloatTensor* arg_running_var = (PyTuple_GET_ITEM(args, 8) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 8)));
      THFloatTensor* arg_save_mean = (PyTuple_GET_ITEM(args, 9) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 9)));
      THFloatTensor* arg_save_std = (PyTuple_GET_ITEM(args, 10) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 10)));
      bool arg_train = (PyTuple_GET_ITEM(args, 11) == Py_True ? true : false);
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 12));
      double arg_eps = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatBatchNormalization_backward(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_gradWeight, arg_gradBias, arg_weight, arg_running_mean, arg_running_var, arg_save_mean, arg_save_std, arg_train, arg_scale, arg_eps);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatBatchNormalization_backward", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, [torch.FloatTensor gradInput or None], [torch.FloatTensor gradWeight or None], [torch.FloatTensor gradBias or None], [torch.FloatTensor weight or None], [torch.FloatTensor running_mean or None], [torch.FloatTensor running_var or None], [torch.FloatTensor save_mean or None], [torch.FloatTensor save_std or None], bool train, float scale, float eps)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleBatchNormalization_backward(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, bool, double, double);

PyObject * DoubleBatchNormalization_backward(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) || PyTuple_GET_ITEM(args, 3) == Py_None) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) || PyTuple_GET_ITEM(args, 5) == Py_None) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) || PyTuple_GET_ITEM(args, 6) == Py_None) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 7)) || PyTuple_GET_ITEM(args, 7) == Py_None) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 8)) || PyTuple_GET_ITEM(args, 8) == Py_None) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 9)) || PyTuple_GET_ITEM(args, 9) == Py_None) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 10)) || PyTuple_GET_ITEM(args, 10) == Py_None) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 11)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 12)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = (PyTuple_GET_ITEM(args, 3) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3)));
      THDoubleTensor* arg_gradWeight = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 5) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5)));
      THDoubleTensor* arg_weight = (PyTuple_GET_ITEM(args, 6) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6)));
      THDoubleTensor* arg_running_mean = (PyTuple_GET_ITEM(args, 7) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 7)));
      THDoubleTensor* arg_running_var = (PyTuple_GET_ITEM(args, 8) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 8)));
      THDoubleTensor* arg_save_mean = (PyTuple_GET_ITEM(args, 9) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 9)));
      THDoubleTensor* arg_save_std = (PyTuple_GET_ITEM(args, 10) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 10)));
      bool arg_train = (PyTuple_GET_ITEM(args, 11) == Py_True ? true : false);
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 12));
      double arg_eps = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleBatchNormalization_backward(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_gradWeight, arg_gradBias, arg_weight, arg_running_mean, arg_running_var, arg_save_mean, arg_save_std, arg_train, arg_scale, arg_eps);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleBatchNormalization_backward", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, [torch.DoubleTensor gradInput or None], [torch.DoubleTensor gradWeight or None], [torch.DoubleTensor gradBias or None], [torch.DoubleTensor weight or None], [torch.DoubleTensor running_mean or None], [torch.DoubleTensor running_var or None], [torch.DoubleTensor save_mean or None], [torch.DoubleTensor save_std or None], bool train, float scale, float eps)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialConvolutionMap_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int);

PyObject * FloatSpatialConvolutionMap_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_connTable = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_nOutputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialConvolutionMap_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_connTable, arg_nInputPlane, arg_nOutputPlane, arg_dW, arg_dH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialConvolutionMap_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, torch.FloatTensor bias, torch.FloatTensor connTable, int nInputPlane, int nOutputPlane, int dW, int dH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialConvolutionMap_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int);

PyObject * DoubleSpatialConvolutionMap_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_connTable = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_nOutputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialConvolutionMap_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_connTable, arg_nInputPlane, arg_nOutputPlane, arg_dW, arg_dH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialConvolutionMap_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, torch.DoubleTensor bias, torch.DoubleTensor connTable, int nInputPlane, int nOutputPlane, int dW, int dH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialConvolutionMap_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int);

PyObject * FloatSpatialConvolutionMap_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_connTable = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_nOutputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialConvolutionMap_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_bias, arg_connTable, arg_nInputPlane, arg_nOutputPlane, arg_dW, arg_dH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialConvolutionMap_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor bias, torch.FloatTensor connTable, int nInputPlane, int nOutputPlane, int dW, int dH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialConvolutionMap_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int);

PyObject * DoubleSpatialConvolutionMap_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_connTable = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_nOutputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialConvolutionMap_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_bias, arg_connTable, arg_nInputPlane, arg_nOutputPlane, arg_dW, arg_dH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialConvolutionMap_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor bias, torch.DoubleTensor connTable, int nInputPlane, int nOutputPlane, int dW, int dH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialConvolutionMap_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, double);

PyObject * FloatSpatialConvolutionMap_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_connTable = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_nOutputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialConvolutionMap_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_connTable, arg_nInputPlane, arg_nOutputPlane, arg_dW, arg_dH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialConvolutionMap_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, torch.FloatTensor connTable, int nInputPlane, int nOutputPlane, int dW, int dH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialConvolutionMap_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, double);

PyObject * DoubleSpatialConvolutionMap_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_connTable = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_nOutputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialConvolutionMap_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_connTable, arg_nInputPlane, arg_nOutputPlane, arg_dW, arg_dH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialConvolutionMap_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, torch.DoubleTensor connTable, int nInputPlane, int nOutputPlane, int dW, int dH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialConvolutionMM_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int);

PyObject * FloatSpatialConvolutionMM_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialConvolutionMM_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialConvolutionMM_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, [torch.FloatTensor bias or None], torch.FloatTensor finput, torch.FloatTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialConvolutionMM_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int);

PyObject * DoubleSpatialConvolutionMM_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialConvolutionMM_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialConvolutionMM_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, [torch.DoubleTensor bias or None], torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialConvolutionMM_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int);

PyObject * FloatSpatialConvolutionMM_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialConvolutionMM_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_fgradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialConvolutionMM_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor finput, torch.FloatTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialConvolutionMM_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int);

PyObject * DoubleSpatialConvolutionMM_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialConvolutionMM_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_fgradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialConvolutionMM_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialConvolutionMM_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, double);

PyObject * FloatSpatialConvolutionMM_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialConvolutionMM_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialConvolutionMM_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, [torch.FloatTensor gradBias or None], torch.FloatTensor finput, torch.FloatTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialConvolutionMM_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, double);

PyObject * DoubleSpatialConvolutionMM_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialConvolutionMM_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialConvolutionMM_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, [torch.DoubleTensor gradBias or None], torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialConvolutionLocal_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int64_t, int64_t, int64_t, int64_t);

PyObject * FloatSpatialConvolutionLocal_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 17 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int64_t arg_inputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int64_t arg_inputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int64_t arg_outputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int64_t arg_outputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialConvolutionLocal_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_inputWidth, arg_inputHeight, arg_outputWidth, arg_outputHeight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialConvolutionLocal_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, torch.FloatTensor bias, torch.FloatTensor finput, torch.FloatTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int inputWidth, int inputHeight, int outputWidth, int outputHeight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialConvolutionLocal_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int64_t, int64_t, int64_t, int64_t);

PyObject * DoubleSpatialConvolutionLocal_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 17 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int64_t arg_inputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int64_t arg_inputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int64_t arg_outputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int64_t arg_outputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialConvolutionLocal_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_inputWidth, arg_inputHeight, arg_outputWidth, arg_outputHeight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialConvolutionLocal_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, torch.DoubleTensor bias, torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int inputWidth, int inputHeight, int outputWidth, int outputHeight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialConvolutionLocal_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int64_t, int64_t, int64_t, int64_t);

PyObject * FloatSpatialConvolutionLocal_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 17 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int64_t arg_inputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int64_t arg_inputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int64_t arg_outputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int64_t arg_outputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialConvolutionLocal_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_fgradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_inputWidth, arg_inputHeight, arg_outputWidth, arg_outputHeight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialConvolutionLocal_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor finput, torch.FloatTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int inputWidth, int inputHeight, int outputWidth, int outputHeight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialConvolutionLocal_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int64_t, int64_t, int64_t, int64_t);

PyObject * DoubleSpatialConvolutionLocal_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 17 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int64_t arg_inputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int64_t arg_inputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int64_t arg_outputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int64_t arg_outputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialConvolutionLocal_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_fgradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_inputWidth, arg_inputHeight, arg_outputWidth, arg_outputHeight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialConvolutionLocal_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int inputWidth, int inputHeight, int outputWidth, int outputHeight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialConvolutionLocal_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int64_t, int64_t, int64_t, int64_t, double);

PyObject * FloatSpatialConvolutionLocal_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 18 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 17))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int64_t arg_inputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int64_t arg_inputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int64_t arg_outputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int64_t arg_outputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 17));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialConvolutionLocal_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_inputWidth, arg_inputHeight, arg_outputWidth, arg_outputHeight, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialConvolutionLocal_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, torch.FloatTensor finput, torch.FloatTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int inputWidth, int inputHeight, int outputWidth, int outputHeight, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialConvolutionLocal_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int64_t, int64_t, int64_t, int64_t, double);

PyObject * DoubleSpatialConvolutionLocal_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 18 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 17))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int64_t arg_inputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int64_t arg_inputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int64_t arg_outputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int64_t arg_outputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 17));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialConvolutionLocal_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_inputWidth, arg_inputHeight, arg_outputWidth, arg_outputHeight, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialConvolutionLocal_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int inputWidth, int inputHeight, int outputWidth, int outputHeight, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialAdaptiveMaxPooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int);

PyObject * FloatSpatialAdaptiveMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialAdaptiveMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_osizeW, arg_osizeH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialAdaptiveMaxPooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.LongTensor indices, int osizeW, int osizeH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialAdaptiveMaxPooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int);

PyObject * DoubleSpatialAdaptiveMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialAdaptiveMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_osizeW, arg_osizeH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialAdaptiveMaxPooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.LongTensor indices, int osizeW, int osizeH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THLongTensor*);

PyObject * FloatSpatialAdaptiveMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialAdaptiveMaxPooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.LongTensor indices)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialAdaptiveMaxPooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THLongTensor*);

PyObject * DoubleSpatialAdaptiveMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialAdaptiveMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialAdaptiveMaxPooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.LongTensor indices)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialAdaptiveAveragePooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int);

PyObject * FloatSpatialAdaptiveAveragePooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialAdaptiveAveragePooling_updateOutput(arg_state, arg_input, arg_output, arg_osizeW, arg_osizeH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialAdaptiveAveragePooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int osizeW, int osizeH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int);

PyObject * DoubleSpatialAdaptiveAveragePooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput(arg_state, arg_input, arg_output, arg_osizeW, arg_osizeH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialAdaptiveAveragePooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int osizeW, int osizeH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatSpatialAdaptiveAveragePooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialAdaptiveAveragePooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleSpatialAdaptiveAveragePooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialAdaptiveAveragePooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialAveragePooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, bool, bool);

PyObject * FloatSpatialAveragePooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 9)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 9) == Py_True ? true : false);
      bool arg_count_include_pad = (PyTuple_GET_ITEM(args, 10) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialAveragePooling_updateOutput(arg_state, arg_input, arg_output, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_ceil_mode, arg_count_include_pad);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialAveragePooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialAveragePooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, bool, bool);

PyObject * DoubleSpatialAveragePooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 9)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 9) == Py_True ? true : false);
      bool arg_count_include_pad = (PyTuple_GET_ITEM(args, 10) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialAveragePooling_updateOutput(arg_state, arg_input, arg_output, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_ceil_mode, arg_count_include_pad);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialAveragePooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialAveragePooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, bool, bool);

PyObject * FloatSpatialAveragePooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 12 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 10)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 11))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 10) == Py_True ? true : false);
      bool arg_count_include_pad = (PyTuple_GET_ITEM(args, 11) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialAveragePooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_ceil_mode, arg_count_include_pad);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialAveragePooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialAveragePooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, bool, bool);

PyObject * DoubleSpatialAveragePooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 12 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 10)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 11))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 10) == Py_True ? true : false);
      bool arg_count_include_pad = (PyTuple_GET_ITEM(args, 11) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialAveragePooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_ceil_mode, arg_count_include_pad);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialAveragePooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialFractionalMaxPooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int, THLongTensor*, THFloatTensor*);

PyObject * FloatSpatialFractionalMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_outputW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_outputH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      THFloatTensor* arg_randomSamples = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialFractionalMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_outputW, arg_outputH, arg_kW, arg_kH, arg_indices, arg_randomSamples);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialFractionalMaxPooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int outputW, int outputH, int kW, int kH, torch.LongTensor indices, torch.FloatTensor randomSamples)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialFractionalMaxPooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, THLongTensor*, THDoubleTensor*);

PyObject * DoubleSpatialFractionalMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 7)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_outputW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_outputH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 7));
      THDoubleTensor* arg_randomSamples = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialFractionalMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_outputW, arg_outputH, arg_kW, arg_kH, arg_indices, arg_randomSamples);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialFractionalMaxPooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int outputW, int outputH, int kW, int kH, torch.LongTensor indices, torch.DoubleTensor randomSamples)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialFractionalMaxPooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, THLongTensor*);

PyObject * FloatSpatialFractionalMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_outputW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_outputH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialFractionalMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_outputW, arg_outputH, arg_kW, arg_kH, arg_indices);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialFractionalMaxPooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int outputW, int outputH, int kW, int kH, torch.LongTensor indices)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialFractionalMaxPooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, THLongTensor*);

PyObject * DoubleSpatialFractionalMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_outputW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_outputH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialFractionalMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_outputW, arg_outputH, arg_kW, arg_kH, arg_indices);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialFractionalMaxPooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int outputW, int outputH, int kW, int kH, torch.LongTensor indices)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialFullConvolution_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int);

PyObject * FloatSpatialFullConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 15 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_columns = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_ones = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_adjW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_adjH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialFullConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_columns, arg_ones, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_adjW, arg_adjH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialFullConvolution_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, [torch.FloatTensor bias or None], torch.FloatTensor columns, torch.FloatTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialFullConvolution_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int);

PyObject * DoubleSpatialFullConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 15 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_columns = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_ones = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_adjW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_adjH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialFullConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_columns, arg_ones, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_adjW, arg_adjH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialFullConvolution_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, [torch.DoubleTensor bias or None], torch.DoubleTensor columns, torch.DoubleTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialFullConvolution_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int);

PyObject * FloatSpatialFullConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_columns = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_adjW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_adjH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialFullConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_columns, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_adjW, arg_adjH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialFullConvolution_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor columns, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialFullConvolution_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int);

PyObject * DoubleSpatialFullConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_columns = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_adjW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_adjH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialFullConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_columns, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_adjW, arg_adjH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialFullConvolution_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor columns, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialFullConvolution_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, double);

PyObject * FloatSpatialFullConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 16 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 15))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_columns = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_ones = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_adjW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_adjH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 15));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialFullConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_columns, arg_ones, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_adjW, arg_adjH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialFullConvolution_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, [torch.FloatTensor gradBias or None], torch.FloatTensor columns, torch.FloatTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialFullConvolution_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, double);

PyObject * DoubleSpatialFullConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 16 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 15))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_columns = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_ones = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_adjW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_adjH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 15));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialFullConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_columns, arg_ones, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_adjW, arg_adjH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialFullConvolution_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, [torch.DoubleTensor gradBias or None], torch.DoubleTensor columns, torch.DoubleTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialFullConvolutionMap_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int);

PyObject * FloatSpatialFullConvolutionMap_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_connTable = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_nOutputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialFullConvolutionMap_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_connTable, arg_nInputPlane, arg_nOutputPlane, arg_dW, arg_dH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialFullConvolutionMap_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, torch.FloatTensor bias, torch.FloatTensor connTable, int nInputPlane, int nOutputPlane, int dW, int dH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialFullConvolutionMap_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int);

PyObject * DoubleSpatialFullConvolutionMap_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_connTable = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_nOutputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialFullConvolutionMap_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_connTable, arg_nInputPlane, arg_nOutputPlane, arg_dW, arg_dH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialFullConvolutionMap_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, torch.DoubleTensor bias, torch.DoubleTensor connTable, int nInputPlane, int nOutputPlane, int dW, int dH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialFullConvolutionMap_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int);

PyObject * FloatSpatialFullConvolutionMap_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_connTable = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_nOutputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialFullConvolutionMap_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_bias, arg_connTable, arg_nInputPlane, arg_nOutputPlane, arg_dW, arg_dH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialFullConvolutionMap_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor bias, torch.FloatTensor connTable, int nInputPlane, int nOutputPlane, int dW, int dH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialFullConvolutionMap_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int);

PyObject * DoubleSpatialFullConvolutionMap_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_connTable = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_nOutputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialFullConvolutionMap_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_bias, arg_connTable, arg_nInputPlane, arg_nOutputPlane, arg_dW, arg_dH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialFullConvolutionMap_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor bias, torch.DoubleTensor connTable, int nInputPlane, int nOutputPlane, int dW, int dH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialFullConvolutionMap_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, double);

PyObject * FloatSpatialFullConvolutionMap_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_connTable = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_nOutputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialFullConvolutionMap_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_connTable, arg_nInputPlane, arg_nOutputPlane, arg_dW, arg_dH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialFullConvolutionMap_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, torch.FloatTensor connTable, int nInputPlane, int nOutputPlane, int dW, int dH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialFullConvolutionMap_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, double);

PyObject * DoubleSpatialFullConvolutionMap_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_connTable = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_nOutputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialFullConvolutionMap_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_connTable, arg_nInputPlane, arg_nOutputPlane, arg_dW, arg_dH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialFullConvolutionMap_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, torch.DoubleTensor connTable, int nInputPlane, int nOutputPlane, int dW, int dH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialDilatedConvolution_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int);

PyObject * FloatSpatialDilatedConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 15 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_columns = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_ones = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialDilatedConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_columns, arg_ones, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialDilatedConvolution_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, [torch.FloatTensor bias or None], torch.FloatTensor columns, torch.FloatTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialDilatedConvolution_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int);

PyObject * DoubleSpatialDilatedConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 15 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_columns = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_ones = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialDilatedConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_columns, arg_ones, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialDilatedConvolution_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, [torch.DoubleTensor bias or None], torch.DoubleTensor columns, torch.DoubleTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialDilatedConvolution_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int);

PyObject * FloatSpatialDilatedConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_columns = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialDilatedConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_columns, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialDilatedConvolution_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor columns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialDilatedConvolution_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int);

PyObject * DoubleSpatialDilatedConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_columns = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialDilatedConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_columns, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialDilatedConvolution_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor columns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialDilatedConvolution_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, double);

PyObject * FloatSpatialDilatedConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 16 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 15))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_columns = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_ones = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 15));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialDilatedConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_columns, arg_ones, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialDilatedConvolution_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, [torch.FloatTensor gradBias or None], torch.FloatTensor columns, torch.FloatTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialDilatedConvolution_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, double);

PyObject * DoubleSpatialDilatedConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 16 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 15))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_columns = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_ones = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 15));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialDilatedConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_columns, arg_ones, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialDilatedConvolution_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, [torch.DoubleTensor gradBias or None], torch.DoubleTensor columns, torch.DoubleTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialFullDilatedConvolution_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int);

PyObject * FloatSpatialFullDilatedConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 17 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_columns = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_ones = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_adjW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_adjH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialFullDilatedConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_columns, arg_ones, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH, arg_adjW, arg_adjH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialFullDilatedConvolution_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, [torch.FloatTensor bias or None], torch.FloatTensor columns, torch.FloatTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialFullDilatedConvolution_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int);

PyObject * DoubleSpatialFullDilatedConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 17 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_columns = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_ones = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_adjW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_adjH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialFullDilatedConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_columns, arg_ones, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH, arg_adjW, arg_adjH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialFullDilatedConvolution_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, [torch.DoubleTensor bias or None], torch.DoubleTensor columns, torch.DoubleTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialFullDilatedConvolution_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int);

PyObject * FloatSpatialFullDilatedConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 16 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_columns = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_adjW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_adjH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialFullDilatedConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_columns, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH, arg_adjW, arg_adjH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialFullDilatedConvolution_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor columns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialFullDilatedConvolution_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int);

PyObject * DoubleSpatialFullDilatedConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 16 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_columns = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_adjW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_adjH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialFullDilatedConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_columns, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH, arg_adjW, arg_adjH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialFullDilatedConvolution_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor columns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialFullDilatedConvolution_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int, double);

PyObject * FloatSpatialFullDilatedConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 18 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 17))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_columns = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_ones = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_adjW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_adjH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 17));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialFullDilatedConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_columns, arg_ones, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH, arg_adjW, arg_adjH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialFullDilatedConvolution_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, [torch.FloatTensor gradBias or None], torch.FloatTensor columns, torch.FloatTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialFullDilatedConvolution_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int, double);

PyObject * DoubleSpatialFullDilatedConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 18 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 17))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_columns = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_ones = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_adjW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_adjH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 17));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialFullDilatedConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_columns, arg_ones, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH, arg_adjW, arg_adjH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialFullDilatedConvolution_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, [torch.DoubleTensor gradBias or None], torch.DoubleTensor columns, torch.DoubleTensor ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialMaxPooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int, int, int, int, int, bool);

PyObject * FloatSpatialMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 10) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_ceil_mode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialMaxPooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.LongTensor indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialMaxPooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int, int, int, int, int, bool);

PyObject * DoubleSpatialMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 10) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_ceil_mode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialMaxPooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.LongTensor indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialMaxPooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int, int, int, int, int, bool);

PyObject * FloatSpatialMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 12 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 11))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 11) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_ceil_mode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialMaxPooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.LongTensor indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialMaxPooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int, int, int, int, int, bool);

PyObject * DoubleSpatialMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 12 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 11))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 11) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_ceil_mode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialMaxPooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.LongTensor indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialDilatedMaxPooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int, int, int, int, int, int, int, bool);

PyObject * FloatSpatialDilatedMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 12) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialDilatedMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH, arg_ceil_mode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialDilatedMaxPooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.LongTensor indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialDilatedMaxPooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int, int, int, int, int, int, int, bool);

PyObject * DoubleSpatialDilatedMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 12) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialDilatedMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH, arg_ceil_mode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialDilatedMaxPooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.LongTensor indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialDilatedMaxPooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int, int, int, int, int, int, int, bool);

PyObject * FloatSpatialDilatedMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 13) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialDilatedMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH, arg_ceil_mode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialDilatedMaxPooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.LongTensor indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialDilatedMaxPooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int, int, int, int, int, int, int, bool);

PyObject * DoubleSpatialDilatedMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 13) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialDilatedMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_dilationW, arg_dilationH, arg_ceil_mode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialDilatedMaxPooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.LongTensor indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialMaxUnpooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int);

PyObject * FloatSpatialMaxUnpooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_owidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_oheight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialMaxUnpooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_owidth, arg_oheight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialMaxUnpooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.LongTensor indices, int owidth, int oheight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialMaxUnpooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int);

PyObject * DoubleSpatialMaxUnpooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_owidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_oheight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialMaxUnpooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_owidth, arg_oheight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialMaxUnpooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.LongTensor indices, int owidth, int oheight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialMaxUnpooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int);

PyObject * FloatSpatialMaxUnpooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_owidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_oheight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialMaxUnpooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_owidth, arg_oheight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialMaxUnpooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.LongTensor indices, int owidth, int oheight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialMaxUnpooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int);

PyObject * DoubleSpatialMaxUnpooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_owidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_oheight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialMaxUnpooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_owidth, arg_oheight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialMaxUnpooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.LongTensor indices, int owidth, int oheight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialSubSampling_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int);

PyObject * FloatSpatialSubSampling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialSubSampling_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_kW, arg_kH, arg_dW, arg_dH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialSubSampling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, torch.FloatTensor bias, int kW, int kH, int dW, int dH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialSubSampling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int);

PyObject * DoubleSpatialSubSampling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialSubSampling_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_kW, arg_kH, arg_dW, arg_dH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialSubSampling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, torch.DoubleTensor bias, int kW, int kH, int dW, int dH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialSubSampling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int);

PyObject * FloatSpatialSubSampling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialSubSampling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_kW, arg_kH, arg_dW, arg_dH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialSubSampling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, int kW, int kH, int dW, int dH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialSubSampling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int);

PyObject * DoubleSpatialSubSampling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialSubSampling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_kW, arg_kH, arg_dW, arg_dH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialSubSampling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, int kW, int kH, int dW, int dH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialSubSampling_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, double);

PyObject * FloatSpatialSubSampling_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialSubSampling_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_kW, arg_kH, arg_dW, arg_dH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialSubSampling_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, torch.FloatTensor gradBias, int kW, int kH, int dW, int dH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialSubSampling_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, double);

PyObject * DoubleSpatialSubSampling_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialSubSampling_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_kW, arg_kH, arg_dW, arg_dH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialSubSampling_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, torch.DoubleTensor gradBias, int kW, int kH, int dW, int dH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialUpSamplingNearest_updateOutput(void*, THFloatTensor*, THFloatTensor*, int);

PyObject * FloatSpatialUpSamplingNearest_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_scale_factor = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialUpSamplingNearest_updateOutput(arg_state, arg_input, arg_output, arg_scale_factor);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialUpSamplingNearest_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int scale_factor)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialUpSamplingNearest_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int);

PyObject * DoubleSpatialUpSamplingNearest_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_scale_factor = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialUpSamplingNearest_updateOutput(arg_state, arg_input, arg_output, arg_scale_factor);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialUpSamplingNearest_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int scale_factor)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialUpSamplingNearest_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int);

PyObject * FloatSpatialUpSamplingNearest_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_scale_factor = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialUpSamplingNearest_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_scale_factor);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialUpSamplingNearest_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int scale_factor)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialUpSamplingNearest_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int);

PyObject * DoubleSpatialUpSamplingNearest_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_scale_factor = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialUpSamplingNearest_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_scale_factor);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialUpSamplingNearest_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int scale_factor)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialUpSamplingBilinear_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int);

PyObject * FloatSpatialUpSamplingBilinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialUpSamplingBilinear_updateOutput(arg_state, arg_input, arg_output, arg_osizeH, arg_osizeW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialUpSamplingBilinear_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int osizeH, int osizeW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialUpSamplingBilinear_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int);

PyObject * DoubleSpatialUpSamplingBilinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialUpSamplingBilinear_updateOutput(arg_state, arg_input, arg_output, arg_osizeH, arg_osizeW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialUpSamplingBilinear_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int osizeH, int osizeW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialUpSamplingBilinear_updateGradInput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int);

PyObject * FloatSpatialUpSamplingBilinear_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_isizeB = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_isizeC = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_isizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_isizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialUpSamplingBilinear_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_isizeB, arg_isizeC, arg_isizeH, arg_isizeW, arg_osizeH, arg_osizeW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialUpSamplingBilinear_updateGradInput", 1, "(int state, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int isizeB, int isizeC, int isizeH, int isizeW, int osizeH, int osizeW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialUpSamplingBilinear_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int);

PyObject * DoubleSpatialUpSamplingBilinear_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_isizeB = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_isizeC = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_isizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_isizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialUpSamplingBilinear_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_isizeB, arg_isizeC, arg_isizeH, arg_isizeW, arg_osizeH, arg_osizeW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialUpSamplingBilinear_updateGradInput", 1, "(int state, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int isizeB, int isizeC, int isizeH, int isizeW, int osizeH, int osizeW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialGridSamplerBilinear_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int);

PyObject * FloatSpatialGridSamplerBilinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_grid = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_padding_mode = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialGridSamplerBilinear_updateOutput(arg_state, arg_input, arg_grid, arg_output, arg_padding_mode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialGridSamplerBilinear_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor grid, torch.FloatTensor output, int padding_mode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialGridSamplerBilinear_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int);

PyObject * DoubleSpatialGridSamplerBilinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_grid = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_padding_mode = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialGridSamplerBilinear_updateOutput(arg_state, arg_input, arg_grid, arg_output, arg_padding_mode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialGridSamplerBilinear_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor grid, torch.DoubleTensor output, int padding_mode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialGridSamplerBilinear_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int);

PyObject * FloatSpatialGridSamplerBilinear_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_grid = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradGrid = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_padding_mode = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialGridSamplerBilinear_updateGradInput(arg_state, arg_input, arg_gradInput, arg_grid, arg_gradGrid, arg_gradOutput, arg_padding_mode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialGridSamplerBilinear_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradInput, torch.FloatTensor grid, torch.FloatTensor gradGrid, torch.FloatTensor gradOutput, int padding_mode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialGridSamplerBilinear_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int);

PyObject * DoubleSpatialGridSamplerBilinear_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_grid = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradGrid = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_padding_mode = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialGridSamplerBilinear_updateGradInput(arg_state, arg_input, arg_gradInput, arg_grid, arg_gradGrid, arg_gradOutput, arg_padding_mode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialGridSamplerBilinear_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradInput, torch.DoubleTensor grid, torch.DoubleTensor gradGrid, torch.DoubleTensor gradOutput, int padding_mode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_Floatunfolded_acc(THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int, int);

PyObject * Floatunfolded_acc(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_inputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_inputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_outputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_Floatunfolded_acc(arg_finput, arg_input, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_nInputPlane, arg_inputWidth, arg_inputHeight, arg_osizeW, arg_outputHeight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "Floatunfolded_acc", 1, "(torch.FloatTensor finput, torch.FloatTensor input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int osizeW, int outputHeight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_Doubleunfolded_acc(THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int, int);

PyObject * Doubleunfolded_acc(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_inputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_inputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_outputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_Doubleunfolded_acc(arg_finput, arg_input, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_nInputPlane, arg_inputWidth, arg_inputHeight, arg_osizeW, arg_outputHeight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "Doubleunfolded_acc", 1, "(torch.DoubleTensor finput, torch.DoubleTensor input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int osizeW, int outputHeight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_Floatunfolded_copy(THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int, int);

PyObject * Floatunfolded_copy(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_inputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_inputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_outputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_outputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_Floatunfolded_copy(arg_finput, arg_input, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_nInputPlane, arg_inputWidth, arg_inputHeight, arg_outputWidth, arg_outputHeight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "Floatunfolded_copy", 1, "(torch.FloatTensor finput, torch.FloatTensor input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_Doubleunfolded_copy(THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int, int);

PyObject * Doubleunfolded_copy(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_nInputPlane = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_inputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_inputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_outputWidth = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_outputHeight = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_Doubleunfolded_copy(arg_finput, arg_input, arg_kW, arg_kH, arg_dW, arg_dH, arg_padW, arg_padH, arg_nInputPlane, arg_inputWidth, arg_inputHeight, arg_outputWidth, arg_outputHeight);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "Doubleunfolded_copy", 1, "(torch.DoubleTensor finput, torch.DoubleTensor input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricAveragePooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, bool, bool);

PyObject * FloatVolumetricAveragePooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 12)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 12) == Py_True ? true : false);
      bool arg_count_include_pad = (PyTuple_GET_ITEM(args, 13) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricAveragePooling_updateOutput(arg_state, arg_input, arg_output, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_padT, arg_padW, arg_padH, arg_ceil_mode, arg_count_include_pad);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricAveragePooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricAveragePooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, bool, bool);

PyObject * DoubleVolumetricAveragePooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 12)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_padT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 12) == Py_True ? true : false);
      bool arg_count_include_pad = (PyTuple_GET_ITEM(args, 13) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricAveragePooling_updateOutput(arg_state, arg_input, arg_output, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_padT, arg_padW, arg_padH, arg_ceil_mode, arg_count_include_pad);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricAveragePooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricAveragePooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, bool, bool);

PyObject * FloatVolumetricAveragePooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 15 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 13)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 14))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 13) == Py_True ? true : false);
      bool arg_count_include_pad = (PyTuple_GET_ITEM(args, 14) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricAveragePooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_padT, arg_padW, arg_padH, arg_ceil_mode, arg_count_include_pad);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricAveragePooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricAveragePooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, bool, bool);

PyObject * DoubleVolumetricAveragePooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 15 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 13)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 14))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_padT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      bool arg_ceil_mode = (PyTuple_GET_ITEM(args, 13) == Py_True ? true : false);
      bool arg_count_include_pad = (PyTuple_GET_ITEM(args, 14) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricAveragePooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_padT, arg_padW, arg_padH, arg_ceil_mode, arg_count_include_pad);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricAveragePooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricConvolution_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int);

PyObject * FloatVolumetricConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricConvolution_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, [torch.FloatTensor bias or None], torch.FloatTensor finput, torch.FloatTensor fgradInput, int dT, int dW, int dH, int pT, int pW, int pH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricConvolution_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int);

PyObject * DoubleVolumetricConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricConvolution_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, [torch.DoubleTensor bias or None], torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int dT, int dW, int dH, int pT, int pW, int pH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricConvolution_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int);

PyObject * FloatVolumetricConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 12 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricConvolution_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor finput, int dT, int dW, int dH, int pT, int pW, int pH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricConvolution_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int);

PyObject * DoubleVolumetricConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 12 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricConvolution_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor finput, int dT, int dW, int dH, int pT, int pW, int pH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricConvolution_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, double);

PyObject * FloatVolumetricConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricConvolution_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, [torch.FloatTensor gradBias or None], torch.FloatTensor finput, torch.FloatTensor fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricConvolution_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, double);

PyObject * DoubleVolumetricConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricConvolution_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, [torch.DoubleTensor gradBias or None], torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricConvolutionMM_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int);

PyObject * FloatVolumetricConvolutionMM_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 16 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricConvolutionMM_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricConvolutionMM_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, [torch.FloatTensor bias or None], torch.FloatTensor finput, torch.FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricConvolutionMM_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int);

PyObject * DoubleVolumetricConvolutionMM_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 16 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricConvolutionMM_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricConvolutionMM_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, [torch.DoubleTensor bias or None], torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricConvolutionMM_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int);

PyObject * FloatVolumetricConvolutionMM_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 16 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricConvolutionMM_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricConvolutionMM_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor finput, torch.FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricConvolutionMM_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int);

PyObject * DoubleVolumetricConvolutionMM_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 16 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricConvolutionMM_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricConvolutionMM_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricConvolutionMM_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, double);

PyObject * FloatVolumetricConvolutionMM_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 17 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 16))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 16));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricConvolutionMM_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricConvolutionMM_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, [torch.FloatTensor gradBias or None], torch.FloatTensor finput, torch.FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricConvolutionMM_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, double);

PyObject * DoubleVolumetricConvolutionMM_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 17 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 16))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 16));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricConvolutionMM_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricConvolutionMM_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, [torch.DoubleTensor gradBias or None], torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricFractionalMaxPooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, THLongTensor*, THFloatTensor*);

PyObject * FloatVolumetricFractionalMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 9)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_outputT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_outputW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_outputH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_poolSizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_poolSizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_poolSizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 9));
      THFloatTensor* arg_randomSamples = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricFractionalMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_outputT, arg_outputW, arg_outputH, arg_poolSizeT, arg_poolSizeW, arg_poolSizeH, arg_indices, arg_randomSamples);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricFractionalMaxPooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, torch.LongTensor indices, torch.FloatTensor randomSamples)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricFractionalMaxPooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, THLongTensor*, THDoubleTensor*);

PyObject * DoubleVolumetricFractionalMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 9)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_outputT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_outputW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_outputH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_poolSizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_poolSizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_poolSizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 9));
      THDoubleTensor* arg_randomSamples = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricFractionalMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_outputT, arg_outputW, arg_outputH, arg_poolSizeT, arg_poolSizeW, arg_poolSizeH, arg_indices, arg_randomSamples);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricFractionalMaxPooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, torch.LongTensor indices, torch.DoubleTensor randomSamples)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricFractionalMaxPooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, THLongTensor*);

PyObject * FloatVolumetricFractionalMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_outputT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_outputW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_outputH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_poolSizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_poolSizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_poolSizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricFractionalMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_outputT, arg_outputW, arg_outputH, arg_poolSizeT, arg_poolSizeW, arg_poolSizeH, arg_indices);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricFractionalMaxPooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, torch.LongTensor indices)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricFractionalMaxPooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, THLongTensor*);

PyObject * DoubleVolumetricFractionalMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_outputT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_outputW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_outputH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_poolSizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_poolSizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_poolSizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricFractionalMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_outputT, arg_outputW, arg_outputH, arg_poolSizeT, arg_poolSizeW, arg_poolSizeH, arg_indices);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricFractionalMaxPooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, torch.LongTensor indices)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricFullConvolution_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int, int, int);

PyObject * FloatVolumetricFullConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 19 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_aT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_aW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_aH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricFullConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_aT, arg_aW, arg_aH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricFullConvolution_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, [torch.FloatTensor bias or None], torch.FloatTensor finput, torch.FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricFullConvolution_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int, int, int);

PyObject * DoubleVolumetricFullConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 19 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_aT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_aW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_aH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricFullConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_aT, arg_aW, arg_aH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricFullConvolution_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, [torch.DoubleTensor bias or None], torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricFullConvolution_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int, int, int);

PyObject * FloatVolumetricFullConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 19 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_aT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_aW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_aH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricFullConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_aT, arg_aW, arg_aH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricFullConvolution_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor finput, torch.FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricFullConvolution_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int, int, int);

PyObject * DoubleVolumetricFullConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 19 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_aT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_aW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_aH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricFullConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_aT, arg_aW, arg_aH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricFullConvolution_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricFullConvolution_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int, int, int, double);

PyObject * FloatVolumetricFullConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 20 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 19))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_aT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_aW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_aH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 19));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricFullConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_aT, arg_aW, arg_aH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricFullConvolution_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, [torch.FloatTensor gradBias or None], torch.FloatTensor finput, torch.FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricFullConvolution_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int, int, int, double);

PyObject * DoubleVolumetricFullConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 20 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 19))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_aT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_aW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_aH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 19));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricFullConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_aT, arg_aW, arg_aH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricFullConvolution_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, [torch.DoubleTensor gradBias or None], torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricDilatedConvolution_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int, int, int);

PyObject * FloatVolumetricDilatedConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 19 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_columns = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_ones = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_padT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricDilatedConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_columns, arg_ones, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_padT, arg_padW, arg_padH, arg_dilationT, arg_dilationW, arg_dilationH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricDilatedConvolution_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, [torch.FloatTensor bias or None], torch.FloatTensor columns, torch.FloatTensor ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricDilatedConvolution_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int, int, int);

PyObject * DoubleVolumetricDilatedConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 19 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_columns = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_ones = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_padT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricDilatedConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_columns, arg_ones, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_padT, arg_padW, arg_padH, arg_dilationT, arg_dilationW, arg_dilationH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricDilatedConvolution_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, [torch.DoubleTensor bias or None], torch.DoubleTensor columns, torch.DoubleTensor ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricDilatedConvolution_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int, int, int);

PyObject * FloatVolumetricDilatedConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 18 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_columns = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricDilatedConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_columns, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_padT, arg_padW, arg_padH, arg_dilationT, arg_dilationW, arg_dilationH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricDilatedConvolution_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor columns, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricDilatedConvolution_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int, int, int);

PyObject * DoubleVolumetricDilatedConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 18 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_columns = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_padT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricDilatedConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_columns, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_padT, arg_padW, arg_padH, arg_dilationT, arg_dilationW, arg_dilationH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricDilatedConvolution_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor columns, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricDilatedConvolution_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int, int, int, double);

PyObject * FloatVolumetricDilatedConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 20 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 19))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_columns = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_ones = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_padT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 19));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricDilatedConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_columns, arg_ones, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_padT, arg_padW, arg_padH, arg_dilationT, arg_dilationW, arg_dilationH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricDilatedConvolution_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, [torch.FloatTensor gradBias or None], torch.FloatTensor columns, torch.FloatTensor ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricDilatedConvolution_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int, int, int, double);

PyObject * DoubleVolumetricDilatedConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 20 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 19))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_columns = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_ones = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_padT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_padW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_padH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 19));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricDilatedConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_columns, arg_ones, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_padT, arg_padW, arg_padH, arg_dilationT, arg_dilationW, arg_dilationH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricDilatedConvolution_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, [torch.DoubleTensor gradBias or None], torch.DoubleTensor columns, torch.DoubleTensor ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricFullDilatedConvolution_updateOutput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int);

PyObject * FloatVolumetricFullDilatedConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 22 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 19)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 20)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 21))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      int arg_aT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 19));
      int arg_aW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 20));
      int arg_aH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 21));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricFullDilatedConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_dilationT, arg_dilationW, arg_dilationH, arg_aT, arg_aW, arg_aH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricFullDilatedConvolution_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor weight, [torch.FloatTensor bias or None], torch.FloatTensor finput, torch.FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricFullDilatedConvolution_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int);

PyObject * DoubleVolumetricFullDilatedConvolution_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 22 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 19)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 20)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 21))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_bias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      int arg_aT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 19));
      int arg_aW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 20));
      int arg_aH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 21));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricFullDilatedConvolution_updateOutput(arg_state, arg_input, arg_output, arg_weight, arg_bias, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_dilationT, arg_dilationW, arg_dilationH, arg_aT, arg_aW, arg_aH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricFullDilatedConvolution_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor weight, [torch.DoubleTensor bias or None], torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricFullDilatedConvolution_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int);

PyObject * FloatVolumetricFullDilatedConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 22 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 19)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 20)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 21))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_weight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      int arg_aT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 19));
      int arg_aW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 20));
      int arg_aH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 21));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricFullDilatedConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_dilationT, arg_dilationW, arg_dilationH, arg_aT, arg_aW, arg_aH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricFullDilatedConvolution_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.FloatTensor weight, torch.FloatTensor finput, torch.FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int);

PyObject * DoubleVolumetricFullDilatedConvolution_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 22 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 19)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 20)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 21))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_weight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      int arg_aT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 19));
      int arg_aW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 20));
      int arg_aH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 21));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_weight, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_dilationT, arg_dilationW, arg_dilationH, arg_aT, arg_aW, arg_aH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricFullDilatedConvolution_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.DoubleTensor weight, torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricFullDilatedConvolution_accGradParameters(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, double);

PyObject * FloatVolumetricFullDilatedConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 23 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 19)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 20)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 21)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 22))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradWeight = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THFloatTensor* arg_finput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THFloatTensor* arg_fgradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      int arg_aT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 19));
      int arg_aW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 20));
      int arg_aH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 21));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 22));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricFullDilatedConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_dilationT, arg_dilationW, arg_dilationH, arg_aT, arg_aW, arg_aH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricFullDilatedConvolution_accGradParameters", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradWeight, [torch.FloatTensor gradBias or None], torch.FloatTensor finput, torch.FloatTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, double);

PyObject * DoubleVolumetricFullDilatedConvolution_accGradParameters(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 23 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          (THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) || PyTuple_GET_ITEM(args, 4) == Py_None) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 5)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 17)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 18)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 19)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 20)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 21)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 22))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradWeight = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradBias = (PyTuple_GET_ITEM(args, 4) == Py_None ? NULL : THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4)));
      THDoubleTensor* arg_finput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 5));
      THDoubleTensor* arg_fgradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 6));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 17));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 18));
      int arg_aT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 19));
      int arg_aW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 20));
      int arg_aH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 21));
      double arg_scale = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 22));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters(arg_state, arg_input, arg_gradOutput, arg_gradWeight, arg_gradBias, arg_finput, arg_fgradInput, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_dilationT, arg_dilationW, arg_dilationH, arg_aT, arg_aW, arg_aH, arg_scale);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricFullDilatedConvolution_accGradParameters", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradWeight, [torch.DoubleTensor gradBias or None], torch.DoubleTensor finput, torch.DoubleTensor fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, float scale)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricMaxPooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int, int, int, int, int, int, int, int, bool);

PyObject * FloatVolumetricMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      bool arg_ceilMode = (PyTuple_GET_ITEM(args, 13) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_ceilMode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricMaxPooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.LongTensor indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricMaxPooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int, int, int, int, int, int, int, int, bool);

PyObject * DoubleVolumetricMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      bool arg_ceilMode = (PyTuple_GET_ITEM(args, 13) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_ceilMode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricMaxPooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.LongTensor indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricMaxPooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int, int, int, int, int, int, int, int, bool);

PyObject * FloatVolumetricMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 15 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 14))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      bool arg_ceilMode = (PyTuple_GET_ITEM(args, 14) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_ceilMode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricMaxPooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.LongTensor indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricMaxPooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int, int, int, int, int, int, int, int, bool);

PyObject * DoubleVolumetricMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 15 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 14))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      bool arg_ceilMode = (PyTuple_GET_ITEM(args, 14) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_ceilMode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricMaxPooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.LongTensor indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricDilatedMaxPooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int, int, int, int, int, int, int, int, int, int, int, bool);

PyObject * FloatVolumetricDilatedMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 17 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 16))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      bool arg_ceilMode = (PyTuple_GET_ITEM(args, 16) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricDilatedMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_dilationT, arg_dilationW, arg_dilationH, arg_ceilMode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricDilatedMaxPooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.LongTensor indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricDilatedMaxPooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int, int, int, int, int, int, int, int, int, int, int, bool);

PyObject * DoubleVolumetricDilatedMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 17 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 16))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      bool arg_ceilMode = (PyTuple_GET_ITEM(args, 16) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricDilatedMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_dilationT, arg_dilationW, arg_dilationH, arg_ceilMode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricDilatedMaxPooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.LongTensor indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricDilatedMaxPooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int, int, int, int, int, int, int, int, int, int, int, bool);

PyObject * FloatVolumetricDilatedMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 18 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 17))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      bool arg_ceilMode = (PyTuple_GET_ITEM(args, 17) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricDilatedMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_dilationT, arg_dilationW, arg_dilationH, arg_ceilMode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricDilatedMaxPooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.LongTensor indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricDilatedMaxPooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int, int, int, int, int, int, int, int, int, int, int, bool);

PyObject * DoubleVolumetricDilatedMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 18 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 14)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 15)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 16)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 17))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_kT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_kW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_kH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      int arg_dilationT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 14));
      int arg_dilationW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 15));
      int arg_dilationH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 16));
      bool arg_ceilMode = (PyTuple_GET_ITEM(args, 17) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricDilatedMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_kT, arg_kW, arg_kH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH, arg_dilationT, arg_dilationW, arg_dilationH, arg_ceilMode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricDilatedMaxPooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.LongTensor indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricMaxUnpooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int, int, int, int, int, int, int, int);

PyObject * FloatVolumetricMaxUnpooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_oT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_oW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_oH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricMaxUnpooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_oT, arg_oW, arg_oH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricMaxUnpooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.LongTensor indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricMaxUnpooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int, int, int, int, int, int, int, int);

PyObject * DoubleVolumetricMaxUnpooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 13 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_oT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_oW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_oH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricMaxUnpooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_oT, arg_oW, arg_oH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricMaxUnpooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.LongTensor indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricMaxUnpooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int, int, int, int, int, int, int, int);

PyObject * FloatVolumetricMaxUnpooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_oT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_oW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_oH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricMaxUnpooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_oT, arg_oW, arg_oH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricMaxUnpooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.LongTensor indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricMaxUnpooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int, int, int, int, int, int, int, int);

PyObject * DoubleVolumetricMaxUnpooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 14 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 11)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 12)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 13))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      int arg_oT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_oW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_oH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_dT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_dW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_dH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      int arg_pT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 11));
      int arg_pW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 12));
      int arg_pH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 13));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricMaxUnpooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices, arg_oT, arg_oW, arg_oH, arg_dT, arg_dW, arg_dH, arg_pT, arg_pW, arg_pH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricMaxUnpooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.LongTensor indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int, int);

PyObject * FloatVolumetricAdaptiveAveragePooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_osizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput(arg_state, arg_input, arg_output, arg_osizeT, arg_osizeW, arg_osizeH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricAdaptiveAveragePooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int osizeT, int osizeW, int osizeH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int);

PyObject * DoubleVolumetricAdaptiveAveragePooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_osizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput(arg_state, arg_input, arg_output, arg_osizeT, arg_osizeW, arg_osizeH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricAdaptiveAveragePooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int osizeT, int osizeW, int osizeH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*);

PyObject * FloatVolumetricAdaptiveAveragePooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricAdaptiveAveragePooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*);

PyObject * DoubleVolumetricAdaptiveAveragePooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricAdaptiveAveragePooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, THLongTensor*, int, int, int);

PyObject * FloatVolumetricAdaptiveMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_osizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_osizeT, arg_osizeW, arg_osizeH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricAdaptiveMaxPooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, torch.LongTensor indices, int osizeT, int osizeW, int osizeH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricAdaptiveMaxPooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, THLongTensor*, int, int, int);

PyObject * DoubleVolumetricAdaptiveMaxPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_osizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricAdaptiveMaxPooling_updateOutput(arg_state, arg_input, arg_output, arg_indices, arg_osizeT, arg_osizeW, arg_osizeH);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricAdaptiveMaxPooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, torch.LongTensor indices, int osizeT, int osizeW, int osizeH)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THLongTensor*);

PyObject * FloatVolumetricAdaptiveMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricAdaptiveMaxPooling_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, torch.LongTensor indices)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricAdaptiveMaxPooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THLongTensor*);

PyObject * DoubleVolumetricAdaptiveMaxPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_LongTensor_Check(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THLongTensor* arg_indices = THNN_LongTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricAdaptiveMaxPooling_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_indices);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricAdaptiveMaxPooling_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, torch.LongTensor indices)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialReflectionPadding_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int);

PyObject * FloatSpatialReflectionPadding_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_top = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_pad_bottom = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialReflectionPadding_updateOutput(arg_state, arg_input, arg_output, arg_pad_left, arg_pad_right, arg_pad_top, arg_pad_bottom);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialReflectionPadding_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int pad_left, int pad_right, int pad_top, int pad_bottom)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialReflectionPadding_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int);

PyObject * DoubleSpatialReflectionPadding_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_top = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_pad_bottom = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialReflectionPadding_updateOutput(arg_state, arg_input, arg_output, arg_pad_left, arg_pad_right, arg_pad_top, arg_pad_bottom);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialReflectionPadding_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int pad_left, int pad_right, int pad_top, int pad_bottom)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialReflectionPadding_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int);

PyObject * FloatSpatialReflectionPadding_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_pad_top = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_pad_bottom = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialReflectionPadding_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_pad_left, arg_pad_right, arg_pad_top, arg_pad_bottom);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialReflectionPadding_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialReflectionPadding_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int);

PyObject * DoubleSpatialReflectionPadding_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_pad_top = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_pad_bottom = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialReflectionPadding_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_pad_left, arg_pad_right, arg_pad_top, arg_pad_bottom);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialReflectionPadding_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialReplicationPadding_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int);

PyObject * FloatSpatialReplicationPadding_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_top = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_pad_bottom = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialReplicationPadding_updateOutput(arg_state, arg_input, arg_output, arg_pad_left, arg_pad_right, arg_pad_top, arg_pad_bottom);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialReplicationPadding_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int pad_left, int pad_right, int pad_top, int pad_bottom)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialReplicationPadding_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int);

PyObject * DoubleSpatialReplicationPadding_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_top = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_pad_bottom = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialReplicationPadding_updateOutput(arg_state, arg_input, arg_output, arg_pad_left, arg_pad_right, arg_pad_top, arg_pad_bottom);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialReplicationPadding_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int pad_left, int pad_right, int pad_top, int pad_bottom)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatSpatialReplicationPadding_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int);

PyObject * FloatSpatialReplicationPadding_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_pad_top = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_pad_bottom = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatSpatialReplicationPadding_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_pad_left, arg_pad_right, arg_pad_top, arg_pad_bottom);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatSpatialReplicationPadding_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleSpatialReplicationPadding_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int);

PyObject * DoubleSpatialReplicationPadding_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 8 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_pad_top = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_pad_bottom = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleSpatialReplicationPadding_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_pad_left, arg_pad_right, arg_pad_top, arg_pad_bottom);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleSpatialReplicationPadding_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatFeatureLPPooling_updateOutput(void*, THFloatTensor*, THFloatTensor*, double, int, int, bool);

PyObject * FloatFeatureLPPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_power = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      int arg_width = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_stride = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      bool arg_batchMode = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatFeatureLPPooling_updateOutput(arg_state, arg_input, arg_output, arg_power, arg_width, arg_stride, arg_batchMode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatFeatureLPPooling_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, float power, int width, int stride, bool batchMode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleFeatureLPPooling_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, double, int, int, bool);

PyObject * DoubleFeatureLPPooling_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 6))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      double arg_power = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 3));
      int arg_width = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_stride = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      bool arg_batchMode = (PyTuple_GET_ITEM(args, 6) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleFeatureLPPooling_updateOutput(arg_state, arg_input, arg_output, arg_power, arg_width, arg_stride, arg_batchMode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleFeatureLPPooling_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, float power, int width, int stride, bool batchMode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatFeatureLPPooling_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, THFloatTensor*, double, int, int, bool);

PyObject * FloatFeatureLPPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      double arg_power = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      int arg_width = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_stride = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      bool arg_batchMode = (PyTuple_GET_ITEM(args, 8) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatFeatureLPPooling_updateGradInput(arg_state, arg_gradOutput, arg_input, arg_output, arg_gradInput, arg_power, arg_width, arg_stride, arg_batchMode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatFeatureLPPooling_updateGradInput", 1, "(int state, torch.FloatTensor gradOutput, torch.FloatTensor input, torch.FloatTensor output, torch.FloatTensor gradInput, float power, int width, int stride, bool batchMode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleFeatureLPPooling_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, double, int, int, bool);

PyObject * DoubleFeatureLPPooling_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 4)) &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          PyBool_Check(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 4));
      double arg_power = THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 5));
      int arg_width = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_stride = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      bool arg_batchMode = (PyTuple_GET_ITEM(args, 8) == Py_True ? true : false);
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleFeatureLPPooling_updateGradInput(arg_state, arg_gradOutput, arg_input, arg_output, arg_gradInput, arg_power, arg_width, arg_stride, arg_batchMode);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleFeatureLPPooling_updateGradInput", 1, "(int state, torch.DoubleTensor gradOutput, torch.DoubleTensor input, torch.DoubleTensor output, torch.DoubleTensor gradInput, float power, int width, int stride, bool batchMode)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricReplicationPadding_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int);

PyObject * FloatVolumetricReplicationPadding_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_top = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_pad_bottom = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_pad_front = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_pad_back = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricReplicationPadding_updateOutput(arg_state, arg_input, arg_output, arg_pad_left, arg_pad_right, arg_pad_top, arg_pad_bottom, arg_pad_front, arg_pad_back);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricReplicationPadding_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_back)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricReplicationPadding_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int);

PyObject * DoubleVolumetricReplicationPadding_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 9 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_top = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_pad_bottom = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_pad_front = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_pad_back = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricReplicationPadding_updateOutput(arg_state, arg_input, arg_output, arg_pad_left, arg_pad_right, arg_pad_top, arg_pad_bottom, arg_pad_front, arg_pad_back);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricReplicationPadding_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_back)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricReplicationPadding_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int);

PyObject * FloatVolumetricReplicationPadding_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_pad_top = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_pad_bottom = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_pad_front = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_pad_back = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricReplicationPadding_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_pad_left, arg_pad_right, arg_pad_top, arg_pad_bottom, arg_pad_front, arg_pad_back);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricReplicationPadding_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_back)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricReplicationPadding_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int);

PyObject * DoubleVolumetricReplicationPadding_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 10 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_pad_top = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_pad_bottom = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_pad_front = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_pad_back = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricReplicationPadding_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_pad_left, arg_pad_right, arg_pad_top, arg_pad_bottom, arg_pad_front, arg_pad_back);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricReplicationPadding_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_back)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricUpSamplingNearest_updateOutput(void*, THFloatTensor*, THFloatTensor*, int);

PyObject * FloatVolumetricUpSamplingNearest_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_scale_factor = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricUpSamplingNearest_updateOutput(arg_state, arg_input, arg_output, arg_scale_factor);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricUpSamplingNearest_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int scale_factor)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricUpSamplingNearest_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int);

PyObject * DoubleVolumetricUpSamplingNearest_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_scale_factor = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricUpSamplingNearest_updateOutput(arg_state, arg_input, arg_output, arg_scale_factor);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricUpSamplingNearest_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int scale_factor)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricUpSamplingNearest_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int);

PyObject * FloatVolumetricUpSamplingNearest_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_scale_factor = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricUpSamplingNearest_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_scale_factor);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricUpSamplingNearest_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int scale_factor)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricUpSamplingNearest_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int);

PyObject * DoubleVolumetricUpSamplingNearest_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_scale_factor = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricUpSamplingNearest_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_scale_factor);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricUpSamplingNearest_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int scale_factor)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricUpSamplingTrilinear_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int, int);

PyObject * FloatVolumetricUpSamplingTrilinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_osizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricUpSamplingTrilinear_updateOutput(arg_state, arg_input, arg_output, arg_osizeT, arg_osizeH, arg_osizeW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricUpSamplingTrilinear_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int osizeT, int osizeH, int osizeW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int);

PyObject * DoubleVolumetricUpSamplingTrilinear_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_osizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput(arg_state, arg_input, arg_output, arg_osizeT, arg_osizeH, arg_osizeW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricUpSamplingTrilinear_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int osizeT, int osizeH, int osizeW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput(void*, THFloatTensor*, THFloatTensor*, int, int, int, int, int, int, int, int);

PyObject * FloatVolumetricUpSamplingTrilinear_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_isizeB = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_isizeC = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_isizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_isizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_isizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_osizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_isizeB, arg_isizeC, arg_isizeT, arg_isizeH, arg_isizeW, arg_osizeT, arg_osizeH, arg_osizeW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatVolumetricUpSamplingTrilinear_updateGradInput", 1, "(int state, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int isizeB, int isizeC, int isizeT, int isizeH, int isizeW, int osizeT, int osizeH, int osizeW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, int, int, int, int, int, int, int, int);

PyObject * DoubleVolumetricUpSamplingTrilinear_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 11 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 7)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 8)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 9)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 10))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_isizeB = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_isizeC = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_isizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      int arg_isizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6));
      int arg_isizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 7));
      int arg_osizeT = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 8));
      int arg_osizeH = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 9));
      int arg_osizeW = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 10));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput(arg_state, arg_gradOutput, arg_gradInput, arg_isizeB, arg_isizeC, arg_isizeT, arg_isizeH, arg_isizeW, arg_osizeT, arg_osizeH, arg_osizeW);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleVolumetricUpSamplingTrilinear_updateGradInput", 1, "(int state, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int isizeB, int isizeC, int isizeT, int isizeH, int isizeW, int osizeT, int osizeH, int osizeW)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalReflectionPadding_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int);

PyObject * FloatTemporalReflectionPadding_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalReflectionPadding_updateOutput(arg_state, arg_input, arg_output, arg_pad_left, arg_pad_right);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalReflectionPadding_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int pad_left, int pad_right)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalReflectionPadding_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int);

PyObject * DoubleTemporalReflectionPadding_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalReflectionPadding_updateOutput(arg_state, arg_input, arg_output, arg_pad_left, arg_pad_right);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalReflectionPadding_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int pad_left, int pad_right)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalReflectionPadding_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int);

PyObject * FloatTemporalReflectionPadding_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalReflectionPadding_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_pad_left, arg_pad_right);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalReflectionPadding_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int pad_left, int pad_right)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalReflectionPadding_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int);

PyObject * DoubleTemporalReflectionPadding_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalReflectionPadding_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_pad_left, arg_pad_right);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalReflectionPadding_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int pad_left, int pad_right)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalReplicationPadding_updateOutput(void*, THFloatTensor*, THFloatTensor*, int, int);

PyObject * FloatTemporalReplicationPadding_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_output = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalReplicationPadding_updateOutput(arg_state, arg_input, arg_output, arg_pad_left, arg_pad_right);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalReplicationPadding_updateOutput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor output, int pad_left, int pad_right)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalReplicationPadding_updateOutput(void*, THDoubleTensor*, THDoubleTensor*, int, int);

PyObject * DoubleTemporalReplicationPadding_updateOutput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_output = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalReplicationPadding_updateOutput(arg_state, arg_input, arg_output, arg_pad_left, arg_pad_right);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalReplicationPadding_updateOutput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor output, int pad_left, int pad_right)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_FloatTemporalReplicationPadding_updateGradInput(void*, THFloatTensor*, THFloatTensor*, THFloatTensor*, int, int);

PyObject * FloatTemporalReplicationPadding_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_FloatTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THFloatTensor* arg_input = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THFloatTensor* arg_gradOutput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THFloatTensor* arg_gradInput = THNN_FloatTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_FloatTemporalReplicationPadding_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_pad_left, arg_pad_right);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "FloatTemporalReplicationPadding_updateGradInput", 1, "(int state, torch.FloatTensor input, torch.FloatTensor gradOutput, torch.FloatTensor gradInput, int pad_left, int pad_right)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    


TH_API void THNN_DoubleTemporalReplicationPadding_updateGradInput(void*, THDoubleTensor*, THDoubleTensor*, THDoubleTensor*, int, int);

PyObject * DoubleTemporalReplicationPadding_updateGradInput(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 1)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 2)) &&
          THNN_DoubleTensor_Check(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {
      
      
      void* arg_state = (void*)THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
      THDoubleTensor* arg_input = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 1));
      THDoubleTensor* arg_gradOutput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 2));
      THDoubleTensor* arg_gradInput = THNN_DoubleTensor_Unpack(PyTuple_GET_ITEM(args, 3));
      int arg_pad_left = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));
      int arg_pad_right = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5));
      
      PyThreadState *_save = NULL;
      try {
        Py_UNBLOCK_THREADS;
        THNN_DoubleTemporalReplicationPadding_updateGradInput(arg_state, arg_input, arg_gradOutput, arg_gradInput, arg_pad_left, arg_pad_right);
        Py_BLOCK_THREADS;
        Py_RETURN_NONE;
      } catch (...) {
        if (_save) {
          Py_BLOCK_THREADS;
        }
        throw;
      }
    
  } else {
    THPUtils_invalidArguments(args, NULL, "DoubleTemporalReplicationPadding_updateGradInput", 1, "(int state, torch.DoubleTensor input, torch.DoubleTensor gradOutput, torch.DoubleTensor gradInput, int pad_left, int pad_right)");
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}
    



static PyMethodDef module_methods[] = {
  {"FloatAbs_updateOutput", (PyCFunction)FloatAbs_updateOutput, METH_VARARGS, NULL},
  {"DoubleAbs_updateOutput", (PyCFunction)DoubleAbs_updateOutput, METH_VARARGS, NULL},
  {"FloatAbs_updateGradInput", (PyCFunction)FloatAbs_updateGradInput, METH_VARARGS, NULL},
  {"DoubleAbs_updateGradInput", (PyCFunction)DoubleAbs_updateGradInput, METH_VARARGS, NULL},
  {"FloatAbsCriterion_updateOutput", (PyCFunction)FloatAbsCriterion_updateOutput, METH_VARARGS, NULL},
  {"DoubleAbsCriterion_updateOutput", (PyCFunction)DoubleAbsCriterion_updateOutput, METH_VARARGS, NULL},
  {"FloatAbsCriterion_updateGradInput", (PyCFunction)FloatAbsCriterion_updateGradInput, METH_VARARGS, NULL},
  {"DoubleAbsCriterion_updateGradInput", (PyCFunction)DoubleAbsCriterion_updateGradInput, METH_VARARGS, NULL},
  {"FloatBCECriterion_updateOutput", (PyCFunction)FloatBCECriterion_updateOutput, METH_VARARGS, NULL},
  {"DoubleBCECriterion_updateOutput", (PyCFunction)DoubleBCECriterion_updateOutput, METH_VARARGS, NULL},
  {"FloatBCECriterion_updateGradInput", (PyCFunction)FloatBCECriterion_updateGradInput, METH_VARARGS, NULL},
  {"DoubleBCECriterion_updateGradInput", (PyCFunction)DoubleBCECriterion_updateGradInput, METH_VARARGS, NULL},
  {"FloatClassNLLCriterion_updateOutput", (PyCFunction)FloatClassNLLCriterion_updateOutput, METH_VARARGS, NULL},
  {"DoubleClassNLLCriterion_updateOutput", (PyCFunction)DoubleClassNLLCriterion_updateOutput, METH_VARARGS, NULL},
  {"FloatClassNLLCriterion_updateGradInput", (PyCFunction)FloatClassNLLCriterion_updateGradInput, METH_VARARGS, NULL},
  {"DoubleClassNLLCriterion_updateGradInput", (PyCFunction)DoubleClassNLLCriterion_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialClassNLLCriterion_updateOutput", (PyCFunction)FloatSpatialClassNLLCriterion_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialClassNLLCriterion_updateOutput", (PyCFunction)DoubleSpatialClassNLLCriterion_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialClassNLLCriterion_updateGradInput", (PyCFunction)FloatSpatialClassNLLCriterion_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialClassNLLCriterion_updateGradInput", (PyCFunction)DoubleSpatialClassNLLCriterion_updateGradInput, METH_VARARGS, NULL},
  {"FloatELU_updateOutput", (PyCFunction)FloatELU_updateOutput, METH_VARARGS, NULL},
  {"DoubleELU_updateOutput", (PyCFunction)DoubleELU_updateOutput, METH_VARARGS, NULL},
  {"FloatELU_updateGradInput", (PyCFunction)FloatELU_updateGradInput, METH_VARARGS, NULL},
  {"DoubleELU_updateGradInput", (PyCFunction)DoubleELU_updateGradInput, METH_VARARGS, NULL},
  {"FloatDistKLDivCriterion_updateOutput", (PyCFunction)FloatDistKLDivCriterion_updateOutput, METH_VARARGS, NULL},
  {"DoubleDistKLDivCriterion_updateOutput", (PyCFunction)DoubleDistKLDivCriterion_updateOutput, METH_VARARGS, NULL},
  {"FloatDistKLDivCriterion_updateGradInput", (PyCFunction)FloatDistKLDivCriterion_updateGradInput, METH_VARARGS, NULL},
  {"DoubleDistKLDivCriterion_updateGradInput", (PyCFunction)DoubleDistKLDivCriterion_updateGradInput, METH_VARARGS, NULL},
  {"FloatGatedLinear_updateOutput", (PyCFunction)FloatGatedLinear_updateOutput, METH_VARARGS, NULL},
  {"DoubleGatedLinear_updateOutput", (PyCFunction)DoubleGatedLinear_updateOutput, METH_VARARGS, NULL},
  {"FloatGatedLinear_updateGradInput", (PyCFunction)FloatGatedLinear_updateGradInput, METH_VARARGS, NULL},
  {"DoubleGatedLinear_updateGradInput", (PyCFunction)DoubleGatedLinear_updateGradInput, METH_VARARGS, NULL},
  {"FloatHardShrink_updateOutput", (PyCFunction)FloatHardShrink_updateOutput, METH_VARARGS, NULL},
  {"DoubleHardShrink_updateOutput", (PyCFunction)DoubleHardShrink_updateOutput, METH_VARARGS, NULL},
  {"FloatHardShrink_updateGradInput", (PyCFunction)FloatHardShrink_updateGradInput, METH_VARARGS, NULL},
  {"DoubleHardShrink_updateGradInput", (PyCFunction)DoubleHardShrink_updateGradInput, METH_VARARGS, NULL},
  {"FloatHardTanh_updateOutput", (PyCFunction)FloatHardTanh_updateOutput, METH_VARARGS, NULL},
  {"DoubleHardTanh_updateOutput", (PyCFunction)DoubleHardTanh_updateOutput, METH_VARARGS, NULL},
  {"FloatHardTanh_updateGradInput", (PyCFunction)FloatHardTanh_updateGradInput, METH_VARARGS, NULL},
  {"DoubleHardTanh_updateGradInput", (PyCFunction)DoubleHardTanh_updateGradInput, METH_VARARGS, NULL},
  {"FloatIm2Col_updateOutput", (PyCFunction)FloatIm2Col_updateOutput, METH_VARARGS, NULL},
  {"DoubleIm2Col_updateOutput", (PyCFunction)DoubleIm2Col_updateOutput, METH_VARARGS, NULL},
  {"FloatIm2Col_updateGradInput", (PyCFunction)FloatIm2Col_updateGradInput, METH_VARARGS, NULL},
  {"DoubleIm2Col_updateGradInput", (PyCFunction)DoubleIm2Col_updateGradInput, METH_VARARGS, NULL},
  {"FloatCol2Im_updateOutput", (PyCFunction)FloatCol2Im_updateOutput, METH_VARARGS, NULL},
  {"DoubleCol2Im_updateOutput", (PyCFunction)DoubleCol2Im_updateOutput, METH_VARARGS, NULL},
  {"FloatCol2Im_updateGradInput", (PyCFunction)FloatCol2Im_updateGradInput, METH_VARARGS, NULL},
  {"DoubleCol2Im_updateGradInput", (PyCFunction)DoubleCol2Im_updateGradInput, METH_VARARGS, NULL},
  {"FloatL1Cost_updateOutput", (PyCFunction)FloatL1Cost_updateOutput, METH_VARARGS, NULL},
  {"DoubleL1Cost_updateOutput", (PyCFunction)DoubleL1Cost_updateOutput, METH_VARARGS, NULL},
  {"FloatL1Cost_updateGradInput", (PyCFunction)FloatL1Cost_updateGradInput, METH_VARARGS, NULL},
  {"DoubleL1Cost_updateGradInput", (PyCFunction)DoubleL1Cost_updateGradInput, METH_VARARGS, NULL},
  {"FloatLeakyReLU_updateOutput", (PyCFunction)FloatLeakyReLU_updateOutput, METH_VARARGS, NULL},
  {"DoubleLeakyReLU_updateOutput", (PyCFunction)DoubleLeakyReLU_updateOutput, METH_VARARGS, NULL},
  {"FloatLeakyReLU_updateGradInput", (PyCFunction)FloatLeakyReLU_updateGradInput, METH_VARARGS, NULL},
  {"DoubleLeakyReLU_updateGradInput", (PyCFunction)DoubleLeakyReLU_updateGradInput, METH_VARARGS, NULL},
  {"FloatGRUFused_updateOutput", (PyCFunction)FloatGRUFused_updateOutput, METH_VARARGS, NULL},
  {"DoubleGRUFused_updateOutput", (PyCFunction)DoubleGRUFused_updateOutput, METH_VARARGS, NULL},
  {"FloatGRUFused_updateGradInput", (PyCFunction)FloatGRUFused_updateGradInput, METH_VARARGS, NULL},
  {"DoubleGRUFused_updateGradInput", (PyCFunction)DoubleGRUFused_updateGradInput, METH_VARARGS, NULL},
  {"FloatLSTMFused_updateOutput", (PyCFunction)FloatLSTMFused_updateOutput, METH_VARARGS, NULL},
  {"DoubleLSTMFused_updateOutput", (PyCFunction)DoubleLSTMFused_updateOutput, METH_VARARGS, NULL},
  {"FloatLSTMFused_updateGradInput", (PyCFunction)FloatLSTMFused_updateGradInput, METH_VARARGS, NULL},
  {"DoubleLSTMFused_updateGradInput", (PyCFunction)DoubleLSTMFused_updateGradInput, METH_VARARGS, NULL},
  {"FloatLogSigmoid_updateOutput", (PyCFunction)FloatLogSigmoid_updateOutput, METH_VARARGS, NULL},
  {"DoubleLogSigmoid_updateOutput", (PyCFunction)DoubleLogSigmoid_updateOutput, METH_VARARGS, NULL},
  {"FloatLogSigmoid_updateGradInput", (PyCFunction)FloatLogSigmoid_updateGradInput, METH_VARARGS, NULL},
  {"DoubleLogSigmoid_updateGradInput", (PyCFunction)DoubleLogSigmoid_updateGradInput, METH_VARARGS, NULL},
  {"FloatLogSoftMax_updateOutput", (PyCFunction)FloatLogSoftMax_updateOutput, METH_VARARGS, NULL},
  {"DoubleLogSoftMax_updateOutput", (PyCFunction)DoubleLogSoftMax_updateOutput, METH_VARARGS, NULL},
  {"FloatLogSoftMax_updateGradInput", (PyCFunction)FloatLogSoftMax_updateGradInput, METH_VARARGS, NULL},
  {"DoubleLogSoftMax_updateGradInput", (PyCFunction)DoubleLogSoftMax_updateGradInput, METH_VARARGS, NULL},
  {"FloatLookupTable_accGradParameters", (PyCFunction)FloatLookupTable_accGradParameters, METH_VARARGS, NULL},
  {"DoubleLookupTable_accGradParameters", (PyCFunction)DoubleLookupTable_accGradParameters, METH_VARARGS, NULL},
  {"FloatLookupTable_renorm", (PyCFunction)FloatLookupTable_renorm, METH_VARARGS, NULL},
  {"DoubleLookupTable_renorm", (PyCFunction)DoubleLookupTable_renorm, METH_VARARGS, NULL},
  {"FloatMarginCriterion_updateOutput", (PyCFunction)FloatMarginCriterion_updateOutput, METH_VARARGS, NULL},
  {"DoubleMarginCriterion_updateOutput", (PyCFunction)DoubleMarginCriterion_updateOutput, METH_VARARGS, NULL},
  {"FloatMarginCriterion_updateGradInput", (PyCFunction)FloatMarginCriterion_updateGradInput, METH_VARARGS, NULL},
  {"DoubleMarginCriterion_updateGradInput", (PyCFunction)DoubleMarginCriterion_updateGradInput, METH_VARARGS, NULL},
  {"FloatSoftMarginCriterion_updateOutput", (PyCFunction)FloatSoftMarginCriterion_updateOutput, METH_VARARGS, NULL},
  {"DoubleSoftMarginCriterion_updateOutput", (PyCFunction)DoubleSoftMarginCriterion_updateOutput, METH_VARARGS, NULL},
  {"FloatSoftMarginCriterion_updateGradInput", (PyCFunction)FloatSoftMarginCriterion_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSoftMarginCriterion_updateGradInput", (PyCFunction)DoubleSoftMarginCriterion_updateGradInput, METH_VARARGS, NULL},
  {"FloatMSECriterion_updateOutput", (PyCFunction)FloatMSECriterion_updateOutput, METH_VARARGS, NULL},
  {"DoubleMSECriterion_updateOutput", (PyCFunction)DoubleMSECriterion_updateOutput, METH_VARARGS, NULL},
  {"FloatMSECriterion_updateGradInput", (PyCFunction)FloatMSECriterion_updateGradInput, METH_VARARGS, NULL},
  {"DoubleMSECriterion_updateGradInput", (PyCFunction)DoubleMSECriterion_updateGradInput, METH_VARARGS, NULL},
  {"FloatMultiLabelMarginCriterion_updateOutput", (PyCFunction)FloatMultiLabelMarginCriterion_updateOutput, METH_VARARGS, NULL},
  {"DoubleMultiLabelMarginCriterion_updateOutput", (PyCFunction)DoubleMultiLabelMarginCriterion_updateOutput, METH_VARARGS, NULL},
  {"FloatMultiLabelMarginCriterion_updateGradInput", (PyCFunction)FloatMultiLabelMarginCriterion_updateGradInput, METH_VARARGS, NULL},
  {"DoubleMultiLabelMarginCriterion_updateGradInput", (PyCFunction)DoubleMultiLabelMarginCriterion_updateGradInput, METH_VARARGS, NULL},
  {"FloatMultiMarginCriterion_updateOutput", (PyCFunction)FloatMultiMarginCriterion_updateOutput, METH_VARARGS, NULL},
  {"DoubleMultiMarginCriterion_updateOutput", (PyCFunction)DoubleMultiMarginCriterion_updateOutput, METH_VARARGS, NULL},
  {"FloatMultiMarginCriterion_updateGradInput", (PyCFunction)FloatMultiMarginCriterion_updateGradInput, METH_VARARGS, NULL},
  {"DoubleMultiMarginCriterion_updateGradInput", (PyCFunction)DoubleMultiMarginCriterion_updateGradInput, METH_VARARGS, NULL},
  {"FloatPReLU_updateOutput", (PyCFunction)FloatPReLU_updateOutput, METH_VARARGS, NULL},
  {"DoublePReLU_updateOutput", (PyCFunction)DoublePReLU_updateOutput, METH_VARARGS, NULL},
  {"FloatPReLU_updateGradInput", (PyCFunction)FloatPReLU_updateGradInput, METH_VARARGS, NULL},
  {"DoublePReLU_updateGradInput", (PyCFunction)DoublePReLU_updateGradInput, METH_VARARGS, NULL},
  {"FloatPReLU_accGradParameters", (PyCFunction)FloatPReLU_accGradParameters, METH_VARARGS, NULL},
  {"DoublePReLU_accGradParameters", (PyCFunction)DoublePReLU_accGradParameters, METH_VARARGS, NULL},
  {"FloatLinear_updateOutput", (PyCFunction)FloatLinear_updateOutput, METH_VARARGS, NULL},
  {"DoubleLinear_updateOutput", (PyCFunction)DoubleLinear_updateOutput, METH_VARARGS, NULL},
  {"FloatLinear_updateGradInput", (PyCFunction)FloatLinear_updateGradInput, METH_VARARGS, NULL},
  {"DoubleLinear_updateGradInput", (PyCFunction)DoubleLinear_updateGradInput, METH_VARARGS, NULL},
  {"FloatLinear_accGradParameters", (PyCFunction)FloatLinear_accGradParameters, METH_VARARGS, NULL},
  {"DoubleLinear_accGradParameters", (PyCFunction)DoubleLinear_accGradParameters, METH_VARARGS, NULL},
  {"FloatRReLU_updateOutput", (PyCFunction)FloatRReLU_updateOutput, METH_VARARGS, NULL},
  {"DoubleRReLU_updateOutput", (PyCFunction)DoubleRReLU_updateOutput, METH_VARARGS, NULL},
  {"FloatRReLU_updateGradInput", (PyCFunction)FloatRReLU_updateGradInput, METH_VARARGS, NULL},
  {"DoubleRReLU_updateGradInput", (PyCFunction)DoubleRReLU_updateGradInput, METH_VARARGS, NULL},
  {"FloatSigmoid_updateOutput", (PyCFunction)FloatSigmoid_updateOutput, METH_VARARGS, NULL},
  {"DoubleSigmoid_updateOutput", (PyCFunction)DoubleSigmoid_updateOutput, METH_VARARGS, NULL},
  {"FloatSigmoid_updateGradInput", (PyCFunction)FloatSigmoid_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSigmoid_updateGradInput", (PyCFunction)DoubleSigmoid_updateGradInput, METH_VARARGS, NULL},
  {"FloatSmoothL1Criterion_updateOutput", (PyCFunction)FloatSmoothL1Criterion_updateOutput, METH_VARARGS, NULL},
  {"DoubleSmoothL1Criterion_updateOutput", (PyCFunction)DoubleSmoothL1Criterion_updateOutput, METH_VARARGS, NULL},
  {"FloatSmoothL1Criterion_updateGradInput", (PyCFunction)FloatSmoothL1Criterion_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSmoothL1Criterion_updateGradInput", (PyCFunction)DoubleSmoothL1Criterion_updateGradInput, METH_VARARGS, NULL},
  {"FloatSoftMax_updateOutput", (PyCFunction)FloatSoftMax_updateOutput, METH_VARARGS, NULL},
  {"DoubleSoftMax_updateOutput", (PyCFunction)DoubleSoftMax_updateOutput, METH_VARARGS, NULL},
  {"FloatSoftMax_updateGradInput", (PyCFunction)FloatSoftMax_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSoftMax_updateGradInput", (PyCFunction)DoubleSoftMax_updateGradInput, METH_VARARGS, NULL},
  {"FloatSoftPlus_updateOutput", (PyCFunction)FloatSoftPlus_updateOutput, METH_VARARGS, NULL},
  {"DoubleSoftPlus_updateOutput", (PyCFunction)DoubleSoftPlus_updateOutput, METH_VARARGS, NULL},
  {"FloatSoftPlus_updateGradInput", (PyCFunction)FloatSoftPlus_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSoftPlus_updateGradInput", (PyCFunction)DoubleSoftPlus_updateGradInput, METH_VARARGS, NULL},
  {"FloatSoftShrink_updateOutput", (PyCFunction)FloatSoftShrink_updateOutput, METH_VARARGS, NULL},
  {"DoubleSoftShrink_updateOutput", (PyCFunction)DoubleSoftShrink_updateOutput, METH_VARARGS, NULL},
  {"FloatSoftShrink_updateGradInput", (PyCFunction)FloatSoftShrink_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSoftShrink_updateGradInput", (PyCFunction)DoubleSoftShrink_updateGradInput, METH_VARARGS, NULL},
  {"FloatIndexLinear_updateOutput", (PyCFunction)FloatIndexLinear_updateOutput, METH_VARARGS, NULL},
  {"DoubleIndexLinear_updateOutput", (PyCFunction)DoubleIndexLinear_updateOutput, METH_VARARGS, NULL},
  {"FloatIndexLinear_accGradParameters", (PyCFunction)FloatIndexLinear_accGradParameters, METH_VARARGS, NULL},
  {"DoubleIndexLinear_accGradParameters", (PyCFunction)DoubleIndexLinear_accGradParameters, METH_VARARGS, NULL},
  {"FloatIndexLinear_accUpdateGradParameters", (PyCFunction)FloatIndexLinear_accUpdateGradParameters, METH_VARARGS, NULL},
  {"DoubleIndexLinear_accUpdateGradParameters", (PyCFunction)DoubleIndexLinear_accUpdateGradParameters, METH_VARARGS, NULL},
  {"FloatIndexLinear_updateParameters", (PyCFunction)FloatIndexLinear_updateParameters, METH_VARARGS, NULL},
  {"DoubleIndexLinear_updateParameters", (PyCFunction)DoubleIndexLinear_updateParameters, METH_VARARGS, NULL},
  {"FloatSparseLinear_updateOutput", (PyCFunction)FloatSparseLinear_updateOutput, METH_VARARGS, NULL},
  {"DoubleSparseLinear_updateOutput", (PyCFunction)DoubleSparseLinear_updateOutput, METH_VARARGS, NULL},
  {"FloatSparseLinear_accGradParameters", (PyCFunction)FloatSparseLinear_accGradParameters, METH_VARARGS, NULL},
  {"DoubleSparseLinear_accGradParameters", (PyCFunction)DoubleSparseLinear_accGradParameters, METH_VARARGS, NULL},
  {"FloatSparseLinear_zeroGradParameters", (PyCFunction)FloatSparseLinear_zeroGradParameters, METH_VARARGS, NULL},
  {"DoubleSparseLinear_zeroGradParameters", (PyCFunction)DoubleSparseLinear_zeroGradParameters, METH_VARARGS, NULL},
  {"FloatSparseLinear_updateParameters", (PyCFunction)FloatSparseLinear_updateParameters, METH_VARARGS, NULL},
  {"DoubleSparseLinear_updateParameters", (PyCFunction)DoubleSparseLinear_updateParameters, METH_VARARGS, NULL},
  {"FloatSparseLinear_legacyUpdateOutput", (PyCFunction)FloatSparseLinear_legacyUpdateOutput, METH_VARARGS, NULL},
  {"DoubleSparseLinear_legacyUpdateOutput", (PyCFunction)DoubleSparseLinear_legacyUpdateOutput, METH_VARARGS, NULL},
  {"FloatSparseLinear_legacyAccGradParameters", (PyCFunction)FloatSparseLinear_legacyAccGradParameters, METH_VARARGS, NULL},
  {"DoubleSparseLinear_legacyAccGradParameters", (PyCFunction)DoubleSparseLinear_legacyAccGradParameters, METH_VARARGS, NULL},
  {"FloatSparseLinear_legacyZeroGradParameters", (PyCFunction)FloatSparseLinear_legacyZeroGradParameters, METH_VARARGS, NULL},
  {"DoubleSparseLinear_legacyZeroGradParameters", (PyCFunction)DoubleSparseLinear_legacyZeroGradParameters, METH_VARARGS, NULL},
  {"FloatSparseLinear_legacyUpdateParameters", (PyCFunction)FloatSparseLinear_legacyUpdateParameters, METH_VARARGS, NULL},
  {"DoubleSparseLinear_legacyUpdateParameters", (PyCFunction)DoubleSparseLinear_legacyUpdateParameters, METH_VARARGS, NULL},
  {"FloatSqrt_updateOutput", (PyCFunction)FloatSqrt_updateOutput, METH_VARARGS, NULL},
  {"DoubleSqrt_updateOutput", (PyCFunction)DoubleSqrt_updateOutput, METH_VARARGS, NULL},
  {"FloatSqrt_updateGradInput", (PyCFunction)FloatSqrt_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSqrt_updateGradInput", (PyCFunction)DoubleSqrt_updateGradInput, METH_VARARGS, NULL},
  {"FloatSquare_updateOutput", (PyCFunction)FloatSquare_updateOutput, METH_VARARGS, NULL},
  {"DoubleSquare_updateOutput", (PyCFunction)DoubleSquare_updateOutput, METH_VARARGS, NULL},
  {"FloatSquare_updateGradInput", (PyCFunction)FloatSquare_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSquare_updateGradInput", (PyCFunction)DoubleSquare_updateGradInput, METH_VARARGS, NULL},
  {"FloatTanh_updateOutput", (PyCFunction)FloatTanh_updateOutput, METH_VARARGS, NULL},
  {"DoubleTanh_updateOutput", (PyCFunction)DoubleTanh_updateOutput, METH_VARARGS, NULL},
  {"FloatTanh_updateGradInput", (PyCFunction)FloatTanh_updateGradInput, METH_VARARGS, NULL},
  {"DoubleTanh_updateGradInput", (PyCFunction)DoubleTanh_updateGradInput, METH_VARARGS, NULL},
  {"FloatThreshold_updateOutput", (PyCFunction)FloatThreshold_updateOutput, METH_VARARGS, NULL},
  {"DoubleThreshold_updateOutput", (PyCFunction)DoubleThreshold_updateOutput, METH_VARARGS, NULL},
  {"FloatThreshold_updateGradInput", (PyCFunction)FloatThreshold_updateGradInput, METH_VARARGS, NULL},
  {"DoubleThreshold_updateGradInput", (PyCFunction)DoubleThreshold_updateGradInput, METH_VARARGS, NULL},
  {"FloatTemporalConvolution_updateOutput", (PyCFunction)FloatTemporalConvolution_updateOutput, METH_VARARGS, NULL},
  {"DoubleTemporalConvolution_updateOutput", (PyCFunction)DoubleTemporalConvolution_updateOutput, METH_VARARGS, NULL},
  {"FloatTemporalConvolution_updateGradInput", (PyCFunction)FloatTemporalConvolution_updateGradInput, METH_VARARGS, NULL},
  {"DoubleTemporalConvolution_updateGradInput", (PyCFunction)DoubleTemporalConvolution_updateGradInput, METH_VARARGS, NULL},
  {"FloatTemporalConvolution_accGradParameters", (PyCFunction)FloatTemporalConvolution_accGradParameters, METH_VARARGS, NULL},
  {"DoubleTemporalConvolution_accGradParameters", (PyCFunction)DoubleTemporalConvolution_accGradParameters, METH_VARARGS, NULL},
  {"FloatTemporalMaxPooling_updateOutput", (PyCFunction)FloatTemporalMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleTemporalMaxPooling_updateOutput", (PyCFunction)DoubleTemporalMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"FloatTemporalMaxPooling_updateGradInput", (PyCFunction)FloatTemporalMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleTemporalMaxPooling_updateGradInput", (PyCFunction)DoubleTemporalMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatTemporalSubSampling_updateOutput", (PyCFunction)FloatTemporalSubSampling_updateOutput, METH_VARARGS, NULL},
  {"DoubleTemporalSubSampling_updateOutput", (PyCFunction)DoubleTemporalSubSampling_updateOutput, METH_VARARGS, NULL},
  {"FloatTemporalSubSampling_updateGradInput", (PyCFunction)FloatTemporalSubSampling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleTemporalSubSampling_updateGradInput", (PyCFunction)DoubleTemporalSubSampling_updateGradInput, METH_VARARGS, NULL},
  {"FloatTemporalSubSampling_accGradParameters", (PyCFunction)FloatTemporalSubSampling_accGradParameters, METH_VARARGS, NULL},
  {"DoubleTemporalSubSampling_accGradParameters", (PyCFunction)DoubleTemporalSubSampling_accGradParameters, METH_VARARGS, NULL},
  {"FloatTemporalRowConvolution_updateOutput", (PyCFunction)FloatTemporalRowConvolution_updateOutput, METH_VARARGS, NULL},
  {"DoubleTemporalRowConvolution_updateOutput", (PyCFunction)DoubleTemporalRowConvolution_updateOutput, METH_VARARGS, NULL},
  {"FloatTemporalRowConvolution_updateGradInput", (PyCFunction)FloatTemporalRowConvolution_updateGradInput, METH_VARARGS, NULL},
  {"DoubleTemporalRowConvolution_updateGradInput", (PyCFunction)DoubleTemporalRowConvolution_updateGradInput, METH_VARARGS, NULL},
  {"FloatTemporalRowConvolution_accGradParameters", (PyCFunction)FloatTemporalRowConvolution_accGradParameters, METH_VARARGS, NULL},
  {"DoubleTemporalRowConvolution_accGradParameters", (PyCFunction)DoubleTemporalRowConvolution_accGradParameters, METH_VARARGS, NULL},
  {"FloatTemporalUpSamplingNearest_updateOutput", (PyCFunction)FloatTemporalUpSamplingNearest_updateOutput, METH_VARARGS, NULL},
  {"DoubleTemporalUpSamplingNearest_updateOutput", (PyCFunction)DoubleTemporalUpSamplingNearest_updateOutput, METH_VARARGS, NULL},
  {"FloatTemporalUpSamplingNearest_updateGradInput", (PyCFunction)FloatTemporalUpSamplingNearest_updateGradInput, METH_VARARGS, NULL},
  {"DoubleTemporalUpSamplingNearest_updateGradInput", (PyCFunction)DoubleTemporalUpSamplingNearest_updateGradInput, METH_VARARGS, NULL},
  {"FloatTemporalUpSamplingLinear_updateOutput", (PyCFunction)FloatTemporalUpSamplingLinear_updateOutput, METH_VARARGS, NULL},
  {"DoubleTemporalUpSamplingLinear_updateOutput", (PyCFunction)DoubleTemporalUpSamplingLinear_updateOutput, METH_VARARGS, NULL},
  {"FloatTemporalUpSamplingLinear_updateGradInput", (PyCFunction)FloatTemporalUpSamplingLinear_updateGradInput, METH_VARARGS, NULL},
  {"DoubleTemporalUpSamplingLinear_updateGradInput", (PyCFunction)DoubleTemporalUpSamplingLinear_updateGradInput, METH_VARARGS, NULL},
  {"FloatBatchNormalization_updateOutput", (PyCFunction)FloatBatchNormalization_updateOutput, METH_VARARGS, NULL},
  {"DoubleBatchNormalization_updateOutput", (PyCFunction)DoubleBatchNormalization_updateOutput, METH_VARARGS, NULL},
  {"FloatBatchNormalization_backward", (PyCFunction)FloatBatchNormalization_backward, METH_VARARGS, NULL},
  {"DoubleBatchNormalization_backward", (PyCFunction)DoubleBatchNormalization_backward, METH_VARARGS, NULL},
  {"FloatSpatialConvolutionMap_updateOutput", (PyCFunction)FloatSpatialConvolutionMap_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialConvolutionMap_updateOutput", (PyCFunction)DoubleSpatialConvolutionMap_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialConvolutionMap_updateGradInput", (PyCFunction)FloatSpatialConvolutionMap_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialConvolutionMap_updateGradInput", (PyCFunction)DoubleSpatialConvolutionMap_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialConvolutionMap_accGradParameters", (PyCFunction)FloatSpatialConvolutionMap_accGradParameters, METH_VARARGS, NULL},
  {"DoubleSpatialConvolutionMap_accGradParameters", (PyCFunction)DoubleSpatialConvolutionMap_accGradParameters, METH_VARARGS, NULL},
  {"FloatSpatialConvolutionMM_updateOutput", (PyCFunction)FloatSpatialConvolutionMM_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialConvolutionMM_updateOutput", (PyCFunction)DoubleSpatialConvolutionMM_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialConvolutionMM_updateGradInput", (PyCFunction)FloatSpatialConvolutionMM_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialConvolutionMM_updateGradInput", (PyCFunction)DoubleSpatialConvolutionMM_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialConvolutionMM_accGradParameters", (PyCFunction)FloatSpatialConvolutionMM_accGradParameters, METH_VARARGS, NULL},
  {"DoubleSpatialConvolutionMM_accGradParameters", (PyCFunction)DoubleSpatialConvolutionMM_accGradParameters, METH_VARARGS, NULL},
  {"FloatSpatialConvolutionLocal_updateOutput", (PyCFunction)FloatSpatialConvolutionLocal_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialConvolutionLocal_updateOutput", (PyCFunction)DoubleSpatialConvolutionLocal_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialConvolutionLocal_updateGradInput", (PyCFunction)FloatSpatialConvolutionLocal_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialConvolutionLocal_updateGradInput", (PyCFunction)DoubleSpatialConvolutionLocal_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialConvolutionLocal_accGradParameters", (PyCFunction)FloatSpatialConvolutionLocal_accGradParameters, METH_VARARGS, NULL},
  {"DoubleSpatialConvolutionLocal_accGradParameters", (PyCFunction)DoubleSpatialConvolutionLocal_accGradParameters, METH_VARARGS, NULL},
  {"FloatSpatialAdaptiveMaxPooling_updateOutput", (PyCFunction)FloatSpatialAdaptiveMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialAdaptiveMaxPooling_updateOutput", (PyCFunction)DoubleSpatialAdaptiveMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialAdaptiveMaxPooling_updateGradInput", (PyCFunction)FloatSpatialAdaptiveMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialAdaptiveMaxPooling_updateGradInput", (PyCFunction)DoubleSpatialAdaptiveMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialAdaptiveAveragePooling_updateOutput", (PyCFunction)FloatSpatialAdaptiveAveragePooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialAdaptiveAveragePooling_updateOutput", (PyCFunction)DoubleSpatialAdaptiveAveragePooling_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialAdaptiveAveragePooling_updateGradInput", (PyCFunction)FloatSpatialAdaptiveAveragePooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialAdaptiveAveragePooling_updateGradInput", (PyCFunction)DoubleSpatialAdaptiveAveragePooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialAveragePooling_updateOutput", (PyCFunction)FloatSpatialAveragePooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialAveragePooling_updateOutput", (PyCFunction)DoubleSpatialAveragePooling_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialAveragePooling_updateGradInput", (PyCFunction)FloatSpatialAveragePooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialAveragePooling_updateGradInput", (PyCFunction)DoubleSpatialAveragePooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialFractionalMaxPooling_updateOutput", (PyCFunction)FloatSpatialFractionalMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialFractionalMaxPooling_updateOutput", (PyCFunction)DoubleSpatialFractionalMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialFractionalMaxPooling_updateGradInput", (PyCFunction)FloatSpatialFractionalMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialFractionalMaxPooling_updateGradInput", (PyCFunction)DoubleSpatialFractionalMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialFullConvolution_updateOutput", (PyCFunction)FloatSpatialFullConvolution_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialFullConvolution_updateOutput", (PyCFunction)DoubleSpatialFullConvolution_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialFullConvolution_updateGradInput", (PyCFunction)FloatSpatialFullConvolution_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialFullConvolution_updateGradInput", (PyCFunction)DoubleSpatialFullConvolution_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialFullConvolution_accGradParameters", (PyCFunction)FloatSpatialFullConvolution_accGradParameters, METH_VARARGS, NULL},
  {"DoubleSpatialFullConvolution_accGradParameters", (PyCFunction)DoubleSpatialFullConvolution_accGradParameters, METH_VARARGS, NULL},
  {"FloatSpatialFullConvolutionMap_updateOutput", (PyCFunction)FloatSpatialFullConvolutionMap_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialFullConvolutionMap_updateOutput", (PyCFunction)DoubleSpatialFullConvolutionMap_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialFullConvolutionMap_updateGradInput", (PyCFunction)FloatSpatialFullConvolutionMap_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialFullConvolutionMap_updateGradInput", (PyCFunction)DoubleSpatialFullConvolutionMap_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialFullConvolutionMap_accGradParameters", (PyCFunction)FloatSpatialFullConvolutionMap_accGradParameters, METH_VARARGS, NULL},
  {"DoubleSpatialFullConvolutionMap_accGradParameters", (PyCFunction)DoubleSpatialFullConvolutionMap_accGradParameters, METH_VARARGS, NULL},
  {"FloatSpatialDilatedConvolution_updateOutput", (PyCFunction)FloatSpatialDilatedConvolution_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialDilatedConvolution_updateOutput", (PyCFunction)DoubleSpatialDilatedConvolution_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialDilatedConvolution_updateGradInput", (PyCFunction)FloatSpatialDilatedConvolution_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialDilatedConvolution_updateGradInput", (PyCFunction)DoubleSpatialDilatedConvolution_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialDilatedConvolution_accGradParameters", (PyCFunction)FloatSpatialDilatedConvolution_accGradParameters, METH_VARARGS, NULL},
  {"DoubleSpatialDilatedConvolution_accGradParameters", (PyCFunction)DoubleSpatialDilatedConvolution_accGradParameters, METH_VARARGS, NULL},
  {"FloatSpatialFullDilatedConvolution_updateOutput", (PyCFunction)FloatSpatialFullDilatedConvolution_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialFullDilatedConvolution_updateOutput", (PyCFunction)DoubleSpatialFullDilatedConvolution_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialFullDilatedConvolution_updateGradInput", (PyCFunction)FloatSpatialFullDilatedConvolution_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialFullDilatedConvolution_updateGradInput", (PyCFunction)DoubleSpatialFullDilatedConvolution_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialFullDilatedConvolution_accGradParameters", (PyCFunction)FloatSpatialFullDilatedConvolution_accGradParameters, METH_VARARGS, NULL},
  {"DoubleSpatialFullDilatedConvolution_accGradParameters", (PyCFunction)DoubleSpatialFullDilatedConvolution_accGradParameters, METH_VARARGS, NULL},
  {"FloatSpatialMaxPooling_updateOutput", (PyCFunction)FloatSpatialMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialMaxPooling_updateOutput", (PyCFunction)DoubleSpatialMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialMaxPooling_updateGradInput", (PyCFunction)FloatSpatialMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialMaxPooling_updateGradInput", (PyCFunction)DoubleSpatialMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialDilatedMaxPooling_updateOutput", (PyCFunction)FloatSpatialDilatedMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialDilatedMaxPooling_updateOutput", (PyCFunction)DoubleSpatialDilatedMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialDilatedMaxPooling_updateGradInput", (PyCFunction)FloatSpatialDilatedMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialDilatedMaxPooling_updateGradInput", (PyCFunction)DoubleSpatialDilatedMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialMaxUnpooling_updateOutput", (PyCFunction)FloatSpatialMaxUnpooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialMaxUnpooling_updateOutput", (PyCFunction)DoubleSpatialMaxUnpooling_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialMaxUnpooling_updateGradInput", (PyCFunction)FloatSpatialMaxUnpooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialMaxUnpooling_updateGradInput", (PyCFunction)DoubleSpatialMaxUnpooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialSubSampling_updateOutput", (PyCFunction)FloatSpatialSubSampling_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialSubSampling_updateOutput", (PyCFunction)DoubleSpatialSubSampling_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialSubSampling_updateGradInput", (PyCFunction)FloatSpatialSubSampling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialSubSampling_updateGradInput", (PyCFunction)DoubleSpatialSubSampling_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialSubSampling_accGradParameters", (PyCFunction)FloatSpatialSubSampling_accGradParameters, METH_VARARGS, NULL},
  {"DoubleSpatialSubSampling_accGradParameters", (PyCFunction)DoubleSpatialSubSampling_accGradParameters, METH_VARARGS, NULL},
  {"FloatSpatialUpSamplingNearest_updateOutput", (PyCFunction)FloatSpatialUpSamplingNearest_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialUpSamplingNearest_updateOutput", (PyCFunction)DoubleSpatialUpSamplingNearest_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialUpSamplingNearest_updateGradInput", (PyCFunction)FloatSpatialUpSamplingNearest_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialUpSamplingNearest_updateGradInput", (PyCFunction)DoubleSpatialUpSamplingNearest_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialUpSamplingBilinear_updateOutput", (PyCFunction)FloatSpatialUpSamplingBilinear_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialUpSamplingBilinear_updateOutput", (PyCFunction)DoubleSpatialUpSamplingBilinear_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialUpSamplingBilinear_updateGradInput", (PyCFunction)FloatSpatialUpSamplingBilinear_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialUpSamplingBilinear_updateGradInput", (PyCFunction)DoubleSpatialUpSamplingBilinear_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialGridSamplerBilinear_updateOutput", (PyCFunction)FloatSpatialGridSamplerBilinear_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialGridSamplerBilinear_updateOutput", (PyCFunction)DoubleSpatialGridSamplerBilinear_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialGridSamplerBilinear_updateGradInput", (PyCFunction)FloatSpatialGridSamplerBilinear_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialGridSamplerBilinear_updateGradInput", (PyCFunction)DoubleSpatialGridSamplerBilinear_updateGradInput, METH_VARARGS, NULL},
  {"Floatunfolded_acc", (PyCFunction)Floatunfolded_acc, METH_VARARGS, NULL},
  {"Doubleunfolded_acc", (PyCFunction)Doubleunfolded_acc, METH_VARARGS, NULL},
  {"Floatunfolded_copy", (PyCFunction)Floatunfolded_copy, METH_VARARGS, NULL},
  {"Doubleunfolded_copy", (PyCFunction)Doubleunfolded_copy, METH_VARARGS, NULL},
  {"FloatVolumetricAveragePooling_updateOutput", (PyCFunction)FloatVolumetricAveragePooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricAveragePooling_updateOutput", (PyCFunction)DoubleVolumetricAveragePooling_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricAveragePooling_updateGradInput", (PyCFunction)FloatVolumetricAveragePooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricAveragePooling_updateGradInput", (PyCFunction)DoubleVolumetricAveragePooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricConvolution_updateOutput", (PyCFunction)FloatVolumetricConvolution_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricConvolution_updateOutput", (PyCFunction)DoubleVolumetricConvolution_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricConvolution_updateGradInput", (PyCFunction)FloatVolumetricConvolution_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricConvolution_updateGradInput", (PyCFunction)DoubleVolumetricConvolution_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricConvolution_accGradParameters", (PyCFunction)FloatVolumetricConvolution_accGradParameters, METH_VARARGS, NULL},
  {"DoubleVolumetricConvolution_accGradParameters", (PyCFunction)DoubleVolumetricConvolution_accGradParameters, METH_VARARGS, NULL},
  {"FloatVolumetricConvolutionMM_updateOutput", (PyCFunction)FloatVolumetricConvolutionMM_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricConvolutionMM_updateOutput", (PyCFunction)DoubleVolumetricConvolutionMM_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricConvolutionMM_updateGradInput", (PyCFunction)FloatVolumetricConvolutionMM_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricConvolutionMM_updateGradInput", (PyCFunction)DoubleVolumetricConvolutionMM_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricConvolutionMM_accGradParameters", (PyCFunction)FloatVolumetricConvolutionMM_accGradParameters, METH_VARARGS, NULL},
  {"DoubleVolumetricConvolutionMM_accGradParameters", (PyCFunction)DoubleVolumetricConvolutionMM_accGradParameters, METH_VARARGS, NULL},
  {"FloatVolumetricFractionalMaxPooling_updateOutput", (PyCFunction)FloatVolumetricFractionalMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricFractionalMaxPooling_updateOutput", (PyCFunction)DoubleVolumetricFractionalMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricFractionalMaxPooling_updateGradInput", (PyCFunction)FloatVolumetricFractionalMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricFractionalMaxPooling_updateGradInput", (PyCFunction)DoubleVolumetricFractionalMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricFullConvolution_updateOutput", (PyCFunction)FloatVolumetricFullConvolution_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricFullConvolution_updateOutput", (PyCFunction)DoubleVolumetricFullConvolution_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricFullConvolution_updateGradInput", (PyCFunction)FloatVolumetricFullConvolution_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricFullConvolution_updateGradInput", (PyCFunction)DoubleVolumetricFullConvolution_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricFullConvolution_accGradParameters", (PyCFunction)FloatVolumetricFullConvolution_accGradParameters, METH_VARARGS, NULL},
  {"DoubleVolumetricFullConvolution_accGradParameters", (PyCFunction)DoubleVolumetricFullConvolution_accGradParameters, METH_VARARGS, NULL},
  {"FloatVolumetricDilatedConvolution_updateOutput", (PyCFunction)FloatVolumetricDilatedConvolution_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricDilatedConvolution_updateOutput", (PyCFunction)DoubleVolumetricDilatedConvolution_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricDilatedConvolution_updateGradInput", (PyCFunction)FloatVolumetricDilatedConvolution_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricDilatedConvolution_updateGradInput", (PyCFunction)DoubleVolumetricDilatedConvolution_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricDilatedConvolution_accGradParameters", (PyCFunction)FloatVolumetricDilatedConvolution_accGradParameters, METH_VARARGS, NULL},
  {"DoubleVolumetricDilatedConvolution_accGradParameters", (PyCFunction)DoubleVolumetricDilatedConvolution_accGradParameters, METH_VARARGS, NULL},
  {"FloatVolumetricFullDilatedConvolution_updateOutput", (PyCFunction)FloatVolumetricFullDilatedConvolution_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricFullDilatedConvolution_updateOutput", (PyCFunction)DoubleVolumetricFullDilatedConvolution_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricFullDilatedConvolution_updateGradInput", (PyCFunction)FloatVolumetricFullDilatedConvolution_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricFullDilatedConvolution_updateGradInput", (PyCFunction)DoubleVolumetricFullDilatedConvolution_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricFullDilatedConvolution_accGradParameters", (PyCFunction)FloatVolumetricFullDilatedConvolution_accGradParameters, METH_VARARGS, NULL},
  {"DoubleVolumetricFullDilatedConvolution_accGradParameters", (PyCFunction)DoubleVolumetricFullDilatedConvolution_accGradParameters, METH_VARARGS, NULL},
  {"FloatVolumetricMaxPooling_updateOutput", (PyCFunction)FloatVolumetricMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricMaxPooling_updateOutput", (PyCFunction)DoubleVolumetricMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricMaxPooling_updateGradInput", (PyCFunction)FloatVolumetricMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricMaxPooling_updateGradInput", (PyCFunction)DoubleVolumetricMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricDilatedMaxPooling_updateOutput", (PyCFunction)FloatVolumetricDilatedMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricDilatedMaxPooling_updateOutput", (PyCFunction)DoubleVolumetricDilatedMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricDilatedMaxPooling_updateGradInput", (PyCFunction)FloatVolumetricDilatedMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricDilatedMaxPooling_updateGradInput", (PyCFunction)DoubleVolumetricDilatedMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricMaxUnpooling_updateOutput", (PyCFunction)FloatVolumetricMaxUnpooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricMaxUnpooling_updateOutput", (PyCFunction)DoubleVolumetricMaxUnpooling_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricMaxUnpooling_updateGradInput", (PyCFunction)FloatVolumetricMaxUnpooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricMaxUnpooling_updateGradInput", (PyCFunction)DoubleVolumetricMaxUnpooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricAdaptiveAveragePooling_updateOutput", (PyCFunction)FloatVolumetricAdaptiveAveragePooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricAdaptiveAveragePooling_updateOutput", (PyCFunction)DoubleVolumetricAdaptiveAveragePooling_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricAdaptiveAveragePooling_updateGradInput", (PyCFunction)FloatVolumetricAdaptiveAveragePooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricAdaptiveAveragePooling_updateGradInput", (PyCFunction)DoubleVolumetricAdaptiveAveragePooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricAdaptiveMaxPooling_updateOutput", (PyCFunction)FloatVolumetricAdaptiveMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricAdaptiveMaxPooling_updateOutput", (PyCFunction)DoubleVolumetricAdaptiveMaxPooling_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricAdaptiveMaxPooling_updateGradInput", (PyCFunction)FloatVolumetricAdaptiveMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricAdaptiveMaxPooling_updateGradInput", (PyCFunction)DoubleVolumetricAdaptiveMaxPooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialReflectionPadding_updateOutput", (PyCFunction)FloatSpatialReflectionPadding_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialReflectionPadding_updateOutput", (PyCFunction)DoubleSpatialReflectionPadding_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialReflectionPadding_updateGradInput", (PyCFunction)FloatSpatialReflectionPadding_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialReflectionPadding_updateGradInput", (PyCFunction)DoubleSpatialReflectionPadding_updateGradInput, METH_VARARGS, NULL},
  {"FloatSpatialReplicationPadding_updateOutput", (PyCFunction)FloatSpatialReplicationPadding_updateOutput, METH_VARARGS, NULL},
  {"DoubleSpatialReplicationPadding_updateOutput", (PyCFunction)DoubleSpatialReplicationPadding_updateOutput, METH_VARARGS, NULL},
  {"FloatSpatialReplicationPadding_updateGradInput", (PyCFunction)FloatSpatialReplicationPadding_updateGradInput, METH_VARARGS, NULL},
  {"DoubleSpatialReplicationPadding_updateGradInput", (PyCFunction)DoubleSpatialReplicationPadding_updateGradInput, METH_VARARGS, NULL},
  {"FloatFeatureLPPooling_updateOutput", (PyCFunction)FloatFeatureLPPooling_updateOutput, METH_VARARGS, NULL},
  {"DoubleFeatureLPPooling_updateOutput", (PyCFunction)DoubleFeatureLPPooling_updateOutput, METH_VARARGS, NULL},
  {"FloatFeatureLPPooling_updateGradInput", (PyCFunction)FloatFeatureLPPooling_updateGradInput, METH_VARARGS, NULL},
  {"DoubleFeatureLPPooling_updateGradInput", (PyCFunction)DoubleFeatureLPPooling_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricReplicationPadding_updateOutput", (PyCFunction)FloatVolumetricReplicationPadding_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricReplicationPadding_updateOutput", (PyCFunction)DoubleVolumetricReplicationPadding_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricReplicationPadding_updateGradInput", (PyCFunction)FloatVolumetricReplicationPadding_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricReplicationPadding_updateGradInput", (PyCFunction)DoubleVolumetricReplicationPadding_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricUpSamplingNearest_updateOutput", (PyCFunction)FloatVolumetricUpSamplingNearest_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricUpSamplingNearest_updateOutput", (PyCFunction)DoubleVolumetricUpSamplingNearest_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricUpSamplingNearest_updateGradInput", (PyCFunction)FloatVolumetricUpSamplingNearest_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricUpSamplingNearest_updateGradInput", (PyCFunction)DoubleVolumetricUpSamplingNearest_updateGradInput, METH_VARARGS, NULL},
  {"FloatVolumetricUpSamplingTrilinear_updateOutput", (PyCFunction)FloatVolumetricUpSamplingTrilinear_updateOutput, METH_VARARGS, NULL},
  {"DoubleVolumetricUpSamplingTrilinear_updateOutput", (PyCFunction)DoubleVolumetricUpSamplingTrilinear_updateOutput, METH_VARARGS, NULL},
  {"FloatVolumetricUpSamplingTrilinear_updateGradInput", (PyCFunction)FloatVolumetricUpSamplingTrilinear_updateGradInput, METH_VARARGS, NULL},
  {"DoubleVolumetricUpSamplingTrilinear_updateGradInput", (PyCFunction)DoubleVolumetricUpSamplingTrilinear_updateGradInput, METH_VARARGS, NULL},
  {"FloatTemporalReflectionPadding_updateOutput", (PyCFunction)FloatTemporalReflectionPadding_updateOutput, METH_VARARGS, NULL},
  {"DoubleTemporalReflectionPadding_updateOutput", (PyCFunction)DoubleTemporalReflectionPadding_updateOutput, METH_VARARGS, NULL},
  {"FloatTemporalReflectionPadding_updateGradInput", (PyCFunction)FloatTemporalReflectionPadding_updateGradInput, METH_VARARGS, NULL},
  {"DoubleTemporalReflectionPadding_updateGradInput", (PyCFunction)DoubleTemporalReflectionPadding_updateGradInput, METH_VARARGS, NULL},
  {"FloatTemporalReplicationPadding_updateOutput", (PyCFunction)FloatTemporalReplicationPadding_updateOutput, METH_VARARGS, NULL},
  {"DoubleTemporalReplicationPadding_updateOutput", (PyCFunction)DoubleTemporalReplicationPadding_updateOutput, METH_VARARGS, NULL},
  {"FloatTemporalReplicationPadding_updateGradInput", (PyCFunction)FloatTemporalReplicationPadding_updateGradInput, METH_VARARGS, NULL},
  {"DoubleTemporalReplicationPadding_updateGradInput", (PyCFunction)DoubleTemporalReplicationPadding_updateGradInput, METH_VARARGS, NULL},

  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION != 2
static struct PyModuleDef module_def = {
   PyModuleDef_HEAD_INIT,
   "torch._thnn._THNN",
   NULL,
   -1,
   module_methods
};
#endif

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_THNN()
#else
PyMODINIT_FUNC PyInit__THNN()
#endif
{
#if PY_MAJOR_VERSION == 2
#define ASSERT_TRUE(cmd) if (!(cmd)) {PyErr_SetString(PyExc_ImportError, "initialization error"); return;}
#else
#define ASSERT_TRUE(cmd) if (!(cmd)) return NULL
#endif
  PyObject *module;

#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("torch._thnn._THNN", module_methods));
#else
  ASSERT_TRUE(module = PyModule_Create(&module_def));
#endif

#if PY_MAJOR_VERSION != 2
  return module;
#endif

#undef ASSERT_TRUE
}
