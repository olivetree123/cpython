#ifndef Py_INTERNAL_CONTEXT_H
#define Py_INTERNAL_CONTEXT_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include "pycore_hamt.h"          // PyHamtObject

#define CONTEXT_MAX_WATCHERS 8

extern PyTypeObject _PyContextTokenMissing_Type;

/* runtime lifecycle */

PyStatus _PyContext_Init(PyInterpreterState *);


/* other API */

// gaojian: _PyContextTokenMissing 是一个单例对象，用于表示缺失的上下文令牌。
typedef struct {
    PyObject_HEAD
} _PyContextTokenMissing;

// gaojian: _pycontextobject 就是 PyContext，是一个上下文对象，用于存储上下文变量
struct _pycontextobject {
    PyObject_HEAD
    // ctx_prev 是一个指向上一个上下文对象的指针。
    PyContext *ctx_prev;
    // ctx_vars 是一个不可变的哈希映射，用于存储上下文变量。
    PyHamtObject *ctx_vars;
    // ctx_weakreflist 是一个弱引用列表，用于存储对此上下文对象的弱引用。
    PyObject *ctx_weakreflist;
    int ctx_entered;
};

// gaojian: _pycontextvarobject 就是 PyContextVar，是一个上下文变量对象，用于存储上下文变量的名称和默认值。
struct _pycontextvarobject {
    PyObject_HEAD
    PyObject *var_name;
    PyObject *var_default;
#ifndef Py_GIL_DISABLED
    PyObject *var_cached;
    uint64_t var_cached_tsid;
    uint64_t var_cached_tsver;
#endif
    Py_hash_t var_hash;
};


// gaojian: _pycontexttokenobject 就是 PyContextToken，是一个上下文令牌对象，用于存储上下文令牌的上下文、上下文变量和旧值。
struct _pycontexttokenobject {
    PyObject_HEAD
    PyContext *tok_ctx;
    PyContextVar *tok_var;
    PyObject *tok_oldval;
    int tok_used;
};


// _testinternalcapi.hamt() used by tests.
// Export for '_testcapi' shared extension
PyAPI_FUNC(PyObject*) _PyContext_NewHamtForTests(void);


#endif /* !Py_INTERNAL_CONTEXT_H */
