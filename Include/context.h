#ifndef Py_CONTEXT_H
#define Py_CONTEXT_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_LIMITED_API


PyAPI_DATA(PyTypeObject) PyContext_Type;
// gaojian: PyContext 是一个上下文对象，用于存储上下文变量
typedef struct _pycontextobject PyContext;

PyAPI_DATA(PyTypeObject) PyContextVar_Type;
// gaojian: PyContextVar 是一个上下文变量对象，用于存储上下文变量的名称和默认值
typedef struct _pycontextvarobject PyContextVar;

PyAPI_DATA(PyTypeObject) PyContextToken_Type;
// gaojian: PyContextToken 是一个上下文令牌对象，用于存储上下文令牌的上下文、上下文变量和旧值
typedef struct _pycontexttokenobject PyContextToken;


#define PyContext_CheckExact(o) Py_IS_TYPE(o, &PyContext_Type)
#define PyContextVar_CheckExact(o) Py_IS_TYPE(o, &PyContextVar_Type)
#define PyContextToken_CheckExact(o) Py_IS_TYPE(o, &PyContextToken_Type)


PyAPI_FUNC(PyObject *) PyContext_New(void);
PyAPI_FUNC(PyObject *) PyContext_Copy(PyObject *);
PyAPI_FUNC(PyObject *) PyContext_CopyCurrent(void);

PyAPI_FUNC(int) PyContext_Enter(PyObject *);
PyAPI_FUNC(int) PyContext_Exit(PyObject *);


/* Create a new context variable.

   default_value can be NULL.

   @gaojian: 创建一个新的上下文变量。
   默认值可以为 NULL。
*/
PyAPI_FUNC(PyObject *) PyContextVar_New(
    const char *name, PyObject *default_value);


/* Get a value for the variable.

   Returns -1 if an error occurred during lookup.

   Returns 0 if value either was or was not found.

   If value was found, *value will point to it.
   If not, it will point to:

   - default_value, if not NULL;
   - the default value of "var", if not NULL;
   - NULL.

   '*value' will be a new ref, if not NULL.

   @gaojian: 获取变量的值。

   如果在查找过程中发生错误，则返回 -1。
   如果找到了值，则返回 0。

   如果找到了值，则 *value 将指向它。
   如果没有找到，则它将指向：
   - default_value，如果不为 NULL；
   - "var" 的默认值，如果不为 NULL；
   - NULL。
   如果不为 NULL，则 '*value' 将是一个新引用。
*/
PyAPI_FUNC(int) PyContextVar_Get(
    PyObject *var, PyObject *default_value, PyObject **value);


/* Set a new value for the variable.
   Returns NULL if an error occurs.
*/
PyAPI_FUNC(PyObject *) PyContextVar_Set(PyObject *var, PyObject *value);


/* Reset a variable to its previous value.
   Returns 0 on success, -1 on error.

   @gaojian: 将变量重置为其先前的值。
   成功时返回 0，错误时返回 -1。
*/
PyAPI_FUNC(int) PyContextVar_Reset(PyObject *var, PyObject *token);


/* This method is exposed only for CPython tests. Don not use it. */
PyAPI_FUNC(PyObject *) _PyContext_NewHamtForTests(void);


#endif /* !Py_LIMITED_API */

#ifdef __cplusplus
}
#endif
#endif /* !Py_CONTEXT_H */
