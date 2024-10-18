"""Support for tasks, coroutines and the scheduler."""

__all__ = (
    'Task', 'create_task',
    'FIRST_COMPLETED', 'FIRST_EXCEPTION', 'ALL_COMPLETED',
    'wait', 'wait_for', 'as_completed', 'sleep',
    'gather', 'shield', 'ensure_future', 'run_coroutine_threadsafe',
    'current_task', 'all_tasks',
    '_register_task', '_unregister_task', '_enter_task', '_leave_task',
)

import concurrent.futures
import contextvars
import functools
import inspect
import itertools
import types
import warnings
import weakref
from types import GenericAlias
from typing import Union, Optional, Coroutine, Generator

from . import base_tasks
from . import coroutines
from . import events
from . import exceptions
from . import futures
from .coroutines import _is_coroutine

# Helper to generate new task names
# This uses itertools.count() instead of a "+= 1" operation because the latter
# is not thread safe. See bpo-11866 for a longer explanation.
_task_name_counter = itertools.count(1).__next__


def current_task(loop=None):
    """Return a currently executed task."""
    if loop is None:
        loop = events.get_running_loop()
    return _current_tasks.get(loop)


def all_tasks(loop=None):
    """Return a set of all tasks for the loop."""
    if loop is None:
        loop = events.get_running_loop()
    # Looping over a WeakSet (_all_tasks) isn't safe as it can be updated from another
    # thread while we do so. Therefore we cast it to list prior to filtering. The list
    # cast itself requires iteration, so we repeat it several times ignoring
    # RuntimeErrors (which are not very likely to occur). See issues 34970 and 36607 for
    # details.
    i = 0
    while True:
        try:
            tasks = list(_all_tasks)
        except RuntimeError:
            i += 1
            if i >= 1000:
                raise
        else:
            break
    return {t for t in tasks
            if futures._get_loop(t) is loop and not t.done()}


def _set_task_name(task, name):
    if name is not None:
        try:
            set_name = task.set_name
        except AttributeError:
            pass
        else:
            set_name(name)


class Task(futures.Future):  # Inherit Python Task implementation
                                # from a Python Future implementation.

    """A coroutine wrapped in a Future."""

    # An important invariant maintained while a Task not done:
    #
    # - Either _fut_waiter is None, and _step() is scheduled;
    # - or _fut_waiter is some Future, and _step() is *not* scheduled.
    #
    # The only transition from the latter to the former is through
    # _wakeup().  When _fut_waiter is not None, one of its callbacks
    # must be _wakeup().

    # If False, don't log a message if the task is destroyed whereas its
    # status is still pending

    # @gaojian:
    # 任务未完成时保持的重要不变量：
    # _fut_waiter 为 None 或 Future。
    # Future 可以是 done() 或 not done()。
    #
    # 任务可以处于以下 3 种状态中的任意一种：
    # - 1：_fut_waiter 不为 None 且不是 _fut_waiter.done()：
    #       __step()未安排，任务正在等待 _fut_waiter。
    # - 2：_fut_waiter 为 None 或 _fut_waiter.done()且 __step() 已安排：
    #       任务正在等待 __step() 执行。
    # - 3：_fut_waiter 为 None 且 __step() *未* 安排：
    #       任务当前正在执行(在 __step()) 中。
    #
    # * 在状态 1 中，__fut_waiter 的回调之一必须是 __wakeup()。
    # * 当 _fut_waiter 变为 done() 时，会发生从 1 到 2 的转换，因为它安排调用 __wakeup()（这会调用 __step()，因此我们知道 __step() 被安排了）。
    # * 当执行 __step() 时，它会从 2 转换为 3，并且它会将_fut_waiter 清除为 None。 
    # 
    # 如果为 False，则在任务被销毁时不会记录消息，而其状态仍处于待处理状态

    _log_destroy_pending = True

    def __init__(self, coro: Union[Coroutine, Generator], *, loop=None, name=None):
        super().__init__(loop=loop)
        if self._source_traceback:
            del self._source_traceback[-1]
        if not coroutines.iscoroutine(coro):
            # raise after Future.__init__(), attrs are required for __del__
            # prevent logging for pending task in __del__
            self._log_destroy_pending = False
            raise TypeError(f"a coroutine was expected, got {coro!r}")

        if name is None:
            self._name = f'Task-{_task_name_counter()}'
        else:
            self._name = str(name)

        self._must_cancel = False
        """这是一个内部属性，用于指示任务或操作是否需要被取消。通过设置这个属性，可以在异步任务中实现任务取消的逻辑。"""

        self._fut_waiter = None
        self._coro: Union[Coroutine, Generator] = coro
        self._context = contextvars.copy_context()

        # @gaojian: 
        # 通过loop.call_sonn，在Task初始化后马上就通知事件循环在下次有空的时候执行自己的__step函数
        self._loop.call_soon(self.__step, context=self._context)
        _register_task(self)

    def __del__(self):
        if self._state == futures._PENDING and self._log_destroy_pending:
            context = {
                'task': self,
                'message': 'Task was destroyed but it is pending!',
            }
            if self._source_traceback:
                context['source_traceback'] = self._source_traceback
            self._loop.call_exception_handler(context)
        super().__del__()

    __class_getitem__ = classmethod(GenericAlias)

    def _repr_info(self):
        return base_tasks._task_repr_info(self)

    def get_coro(self):
        return self._coro

    def get_name(self):
        return self._name

    def set_name(self, value):
        self._name = str(value)

    def set_result(self, result):
        raise RuntimeError('Task does not support set_result operation')

    def set_exception(self, exception):
        raise RuntimeError('Task does not support set_exception operation')

    def get_stack(self, *, limit=None):
        """Return the list of stack frames for this task's coroutine.

        If the coroutine is not done, this returns the stack where it is
        suspended.  If the coroutine has completed successfully or was
        cancelled, this returns an empty list.  If the coroutine was
        terminated by an exception, this returns the list of traceback
        frames.

        The frames are always ordered from oldest to newest.

        The optional limit gives the maximum number of frames to
        return; by default all available frames are returned.  Its
        meaning differs depending on whether a stack or a traceback is
        returned: the newest frames of a stack are returned, but the
        oldest frames of a traceback are returned.  (This matches the
        behavior of the traceback module.)

        For reasons beyond our control, only one stack frame is
        returned for a suspended coroutine.
        """
        return base_tasks._task_get_stack(self, limit)

    def print_stack(self, *, limit=None, file=None):
        """Print the stack or traceback for this task's coroutine.

        This produces output similar to that of the traceback module,
        for the frames retrieved by get_stack().  The limit argument
        is passed to get_stack().  The file argument is an I/O stream
        to which the output is written; by default output is written
        to sys.stderr.
        """
        return base_tasks._task_print_stack(self, limit, file)

    def cancel(self, msg=None):
        """Request that this task cancel itself.

        This arranges for a CancelledError to be thrown into the
        wrapped coroutine on the next cycle through the event loop.
        The coroutine then has a chance to clean up or even deny
        the request using try/except/finally.

        Unlike Future.cancel, this does not guarantee that the
        task will be cancelled: the exception might be caught and
        acted upon, delaying cancellation of the task or preventing
        cancellation completely.  The task may also return a value or
        raise a different exception.

        Immediately after this method is called, Task.cancelled() will
        not return True (unless the task was already cancelled).  A
        task will be marked as cancelled when the wrapped coroutine
        terminates with a CancelledError exception (even if cancel()
        was not called).
        """
        self._log_traceback = False
        if self.done():
            return False
        if self._fut_waiter is not None:
            if self._fut_waiter.cancel(msg=msg):
                # Leave self._fut_waiter; it may be a Task that
                # catches and ignores the cancellation so we may have
                # to cancel it again later.
                return True
        # It must be the case that self.__step is already scheduled.
        self._must_cancel = True
        self._cancel_message = msg
        return True

    def __step(self, exc=None):
        if self.done():
            raise exceptions.InvalidStateError(
                f'_step(): already done: {self!r}, {exc!r}')
        if self._must_cancel:
            if not isinstance(exc, exceptions.CancelledError):
                exc = self._make_cancelled_error()
            self._must_cancel = False
        coro = self._coro
        self._fut_waiter = None

        _enter_task(self._loop, self)
        # Call either coro.throw(exc) or coro.send(None).
        try:
            if exc is None:
                # We use the `send` method directly, because coroutines
                # don't have `__iter__` and `__next__` methods.

                # @gaojian:
                # 这行代码是用来启动一个协程（coroutine）的。具体来说，它调用协程的send(None)方法，并接收协程的第一个 `yield` 表达式的结果。以下是详细解释：
                # 1. `coro`是一个协程对象，通常是通过调用一个带有 `yield` 关键字的生成器函数得到的；
                # 2. `send(None)`是启动协程的标准方式，相当于调用 `next(coro)`；
                # 3. `result`将接收协程的第一个 `yield` 表达式的值；
                # 简而言之，这行代码的作用是启动协程并获取其第一个 `yield` 表达式的结果。
                result: Union[Task, futures.Future] = coro.send(None)
            else:
                result = coro.throw(exc)
        except StopIteration as exc:
            if self._must_cancel:
                # Task is cancelled right before coro stops.
                self._must_cancel = False
                super().cancel(msg=self._cancel_message)
            else:
                super().set_result(exc.value)
        except exceptions.CancelledError as exc:
            # Save the original exception so we can chain it later.
            self._cancelled_exc = exc
            super().cancel()  # I.e., Future.cancel(self).
        except (KeyboardInterrupt, SystemExit) as exc:
            super().set_exception(exc)
            raise
        except BaseException as exc:
            super().set_exception(exc)
        else:
            # 没有异常
            # 检查当前任务的结果result有没有 _asyncio_future_blocking 属性：
            # - 如果有，说明这个result是一个Future或Task对象，也就是说当前任务里面还有一个子任务(一个Future或Task对象)需要等待
            #
            # result 只能是Future对象或者None
            blocking: Optional[bool] = getattr(result, '_asyncio_future_blocking', None)
            if blocking is not None:
                # Yielded Future must come from Future.__iter__().
                if futures._get_loop(result) is not self._loop:
                    new_exc = RuntimeError(f'Task {self!r} got Future {result!r} attached to a different loop')
                    self._loop.call_soon(self.__step, new_exc, context=self._context)
                elif blocking:
                    # blocking = True 表示这个子任务处于正在执行的状态
                    # 此时的Task放弃了自己对事件循环的控制权，等待这个正在执行的子任务执行完成后唤醒自己
                    # 子任务执行完成后会调用 父任务的__wakeup() 方法唤醒父任务
                    result._asyncio_future_blocking = False
                    result.add_done_callback(self.__wakeup, context=self._context)
                    self._fut_waiter = result
                    # if self._must_cancel:
                    #     if self._fut_waiter.cancel(msg=self._cancel_message):
                    #         self._must_cancel = False
                else:
                    new_exc = RuntimeError(
                        f'yield was used instead of yield from '
                        f'in task {self!r} with {result!r}')
                    self._loop.call_soon(
                        self.__step, new_exc, context=self._context)

            elif result is None:
                # Bare yield relinquishes control for one event loop iteration.
                # 一个空的 yield 表达式会让出控制权，等待下一个事件循环迭代
                self._loop.call_soon(self.__step, context=self._context)
            elif inspect.isgenerator(result):
                # Yielding a generator is just wrong.
                new_exc = RuntimeError(
                    f'yield was used instead of yield from for '
                    f'generator in task {self!r} with {result!r}')
                self._loop.call_soon(
                    self.__step, new_exc, context=self._context)
            else:
                # Yielding something else is an error.
                new_exc = RuntimeError(f'Task got bad yield: {result!r}')
                self._loop.call_soon(
                    self.__step, new_exc, context=self._context)
        finally:
            _leave_task(self._loop, self)
            self = None  # Needed to break cycles when an exception occurs.

    def __wakeup(self, future: Union["Task", futures.Future]):
        try:
            future.result()
        except BaseException as exc:
            # This may also be a cancellation.
            self.__step(exc)
        else:
            # Don't pass the value of `future.result()` explicitly,
            # as `Future.__iter__` and `Future.__await__` don't need it.
            # If we call `_step(value, None)` instead of `_step()`,
            # Python eval loop would use `.send(value)` method call,
            # instead of `__next__()`, which is slower for futures
            # that return non-generator iterators from their `__iter__`.

            # @gaojian:
            # 不要显式传递 `future.result()` 的值，
            # 因为 `Future.__iter__` 和 `Future.__await__` 不需要它。
            # 如果我们调用 `_step(value, None)` 而不是 `_step()`，
            # Python eval 循环将使用 `.send(value)` 方法调用，
            # 而不是 `__next__()`，对于 future 来说从它们的`__iter__`返回非生成器迭代器速度较慢。
            self.__step()
        self = None  # Needed to break cycles when an exception occurs.


_PyTask = Task


try:
    import _asyncio
except ImportError:
    pass
else:
    # _CTask is needed for tests.
    Task = _CTask = _asyncio.Task


def create_task(coro, *, name=None):
    """Schedule the execution of a coroutine object in a spawn task.

    Return a Task object.
    """
    loop = events.get_running_loop()
    task = loop.create_task(coro)
    _set_task_name(task, name)
    return task


# wait() and as_completed() similar to those in PEP 3148.

FIRST_COMPLETED = concurrent.futures.FIRST_COMPLETED
FIRST_EXCEPTION = concurrent.futures.FIRST_EXCEPTION
ALL_COMPLETED = concurrent.futures.ALL_COMPLETED


async def wait(fs, *, timeout=None, return_when=ALL_COMPLETED):
    """Wait for the Futures and coroutines given by fs to complete.

    The fs iterable must not be empty.

    Coroutines will be wrapped in Tasks.

    Returns two sets of Future: (done, pending).

    Usage:

        done, pending = await asyncio.wait(fs)

    Note: This does not raise TimeoutError! Futures that aren't done
    when the timeout occurs are returned in the second set.
    """
    if futures.isfuture(fs) or coroutines.iscoroutine(fs):
        raise TypeError(f"expect a list of futures, not {type(fs).__name__}")
    if not fs:
        raise ValueError('Set of coroutines/Futures is empty.')
    if return_when not in (FIRST_COMPLETED, FIRST_EXCEPTION, ALL_COMPLETED):
        raise ValueError(f'Invalid return_when value: {return_when}')

    loop = events.get_running_loop()

    fs = set(fs)

    if any(coroutines.iscoroutine(f) for f in fs):
        warnings.warn("The explicit passing of coroutine objects to "
                      "asyncio.wait() is deprecated since Python 3.8, and "
                      "scheduled for removal in Python 3.11.",
                      DeprecationWarning, stacklevel=2)

    fs = {ensure_future(f, loop=loop) for f in fs}

    return await _wait(fs, timeout, return_when, loop)


def _release_waiter(waiter, *args):
    if not waiter.done():
        waiter.set_result(None)


async def wait_for(fut, timeout):
    """Wait for the single Future or coroutine to complete, with timeout.

    Coroutine will be wrapped in Task.

    Returns result of the Future or coroutine.  When a timeout occurs,
    it cancels the task and raises TimeoutError.  To avoid the task
    cancellation, wrap it in shield().

    If the wait is cancelled, the task is also cancelled.

    This function is a coroutine.
    """
    loop = events.get_running_loop()

    if timeout is None:
        return await fut

    if timeout <= 0:
        fut = ensure_future(fut, loop=loop)

        if fut.done():
            return fut.result()

        await _cancel_and_wait(fut, loop=loop)
        try:
            return fut.result()
        except exceptions.CancelledError as exc:
            raise exceptions.TimeoutError() from exc

    waiter = loop.create_future()
    timeout_handle = loop.call_later(timeout, _release_waiter, waiter)
    cb = functools.partial(_release_waiter, waiter)

    fut = ensure_future(fut, loop=loop)
    fut.add_done_callback(cb)

    try:
        # wait until the future completes or the timeout
        try:
            await waiter
        except exceptions.CancelledError:
            if fut.done():
                return fut.result()
            else:
                fut.remove_done_callback(cb)
                # We must ensure that the task is not running
                # after wait_for() returns.
                # See https://bugs.python.org/issue32751
                await _cancel_and_wait(fut, loop=loop)
                raise

        if fut.done():
            return fut.result()
        else:
            fut.remove_done_callback(cb)
            # We must ensure that the task is not running
            # after wait_for() returns.
            # See https://bugs.python.org/issue32751
            await _cancel_and_wait(fut, loop=loop)
            # In case task cancellation failed with some
            # exception, we should re-raise it
            # See https://bugs.python.org/issue40607
            try:
                return fut.result()
            except exceptions.CancelledError as exc:
                raise exceptions.TimeoutError() from exc
    finally:
        timeout_handle.cancel()


async def _wait(fs: list[futures.Future], timeout, return_when, loop):
    """Internal helper for wait().

    The fs argument must be a collection of Futures.
    """
    assert fs, 'Set of Futures is empty.'
    waiter = loop.create_future()
    timeout_handle = None
    if timeout is not None:
        timeout_handle = loop.call_later(timeout, _release_waiter, waiter)
    counter = len(fs)

    def _on_completion(f):
        nonlocal counter
        counter -= 1
        if (counter <= 0 or
            return_when == FIRST_COMPLETED or
            return_when == FIRST_EXCEPTION and (not f.cancelled() and
                                                f.exception() is not None)):
            if timeout_handle is not None:
                timeout_handle.cancel()
            if not waiter.done():
                waiter.set_result(None)

    for f in fs:
        f.add_done_callback(_on_completion)

    try:
        await waiter
    finally:
        if timeout_handle is not None:
            timeout_handle.cancel()
        for f in fs:
            f.remove_done_callback(_on_completion)

    done, pending = set(), set()
    for f in fs:
        if f.done():
            done.add(f)
        else:
            pending.add(f)
    return done, pending


async def _cancel_and_wait(fut, loop):
    """Cancel the *fut* future or task and wait until it completes."""

    waiter = loop.create_future()
    cb = functools.partial(_release_waiter, waiter)
    fut.add_done_callback(cb)

    try:
        fut.cancel()
        # We cannot wait on *fut* directly to make
        # sure _cancel_and_wait itself is reliably cancellable.
        await waiter
    finally:
        fut.remove_done_callback(cb)


# This is *not* a @coroutine!  It is just an iterator (yielding Futures).
def as_completed(fs, *, timeout=None):
    """Return an iterator whose values are coroutines.

    When waiting for the yielded coroutines you'll get the results (or
    exceptions!) of the original Futures (or coroutines), in the order
    in which and as soon as they complete.

    This differs from PEP 3148; the proper way to use this is:

        for f in as_completed(fs):
            result = await f  # The 'await' may raise.
            # Use result.

    If a timeout is specified, the 'await' will raise
    TimeoutError when the timeout occurs before all Futures are done.

    Note: The futures 'f' are not necessarily members of fs.

    @gaojian: 按完成顺序创建可等待项或其结果的迭代器。
    """
    if futures.isfuture(fs) or coroutines.iscoroutine(fs):
        raise TypeError(f"expect an iterable of futures, not {type(fs).__name__}")

    from .queues import Queue  # Import here to avoid circular import problem.
    done = Queue()

    loop = events._get_event_loop()
    todo = {ensure_future(f, loop=loop) for f in set(fs)}
    timeout_handle = None

    def _on_timeout():
        for f in todo:
            f.remove_done_callback(_on_completion)
            done.put_nowait(None)  # Queue a dummy value for _wait_for_one().
        todo.clear()  # Can't do todo.remove(f) in the loop.

    def _on_completion(f):
        if not todo:
            return  # _on_timeout() was here first.
        todo.remove(f)
        done.put_nowait(f)
        if not todo and timeout_handle is not None:
            timeout_handle.cancel()

    async def _wait_for_one():
        f = await done.get()
        if f is None:
            # Dummy value from _on_timeout().
            raise exceptions.TimeoutError
        return f.result()  # May raise f.exception().

    for f in todo:
        f.add_done_callback(_on_completion)
    if todo and timeout is not None:
        timeout_handle = loop.call_later(timeout, _on_timeout)
    for _ in range(len(todo)):
        yield _wait_for_one()


@types.coroutine
def __sleep0():
    """Skip one event loop run cycle.

    This is a private helper for 'asyncio.sleep()', used
    when the 'delay' is set to 0.  It uses a bare 'yield'
    expression (which Task.__step knows how to handle)
    instead of creating a Future object.
    """
    yield


async def sleep(delay, result=None):
    """Coroutine that completes after a given time (in seconds)."""
    if delay <= 0:
        await __sleep0()
        return result

    loop = events.get_running_loop()
    future = loop.create_future()
    h = loop.call_later(delay,
                        futures._set_result_unless_cancelled,
                        future, result)
    try:
        return await future
    finally:
        h.cancel()


def ensure_future(coro_or_future, *, loop=None):
    """Wrap a coroutine or an awaitable in a future.

    If the argument is a Future, it is returned directly.

    @gaojian
    ensure_future() 函数用来将一个协程或可等待对象包装在一个 future 中。
    - 如果参数是一个 Future，则直接返回；
    - 如果参数是一个协程，则将其包装在一个 Task 中；
    """
    return _ensure_future(coro_or_future, loop=loop)


def _ensure_future(coro_or_future, *, loop=None):
    if futures.isfuture(coro_or_future):
        if loop is not None and loop is not futures._get_loop(coro_or_future):
            raise ValueError('The future belongs to a different loop than '
                            'the one specified as the loop argument')
        return coro_or_future
    called_wrap_awaitable = False
    if not coroutines.iscoroutine(coro_or_future):
        if inspect.isawaitable(coro_or_future):
            coro_or_future = _wrap_awaitable(coro_or_future)
            called_wrap_awaitable = True
        else:
            raise TypeError('An asyncio.Future, a coroutine or an awaitable '
                            'is required')

    if loop is None:
        loop = events._get_event_loop(stacklevel=4)
    try:
        return loop.create_task(coro_or_future)
    except RuntimeError: 
        if not called_wrap_awaitable:
            coro_or_future.close()
        raise


@types.coroutine
def _wrap_awaitable(awaitable):
    """Helper for asyncio.ensure_future().

    Wraps awaitable (an object with __await__) into a coroutine
    that will later be wrapped in a Task by ensure_future().
    """
    return (yield from awaitable.__await__())

_wrap_awaitable._is_coroutine = _is_coroutine


class _GatheringFuture(futures.Future):
    """Helper for gather().

    This overrides cancel() to cancel all the children and act more
    like Task.cancel(), which doesn't immediately mark itself as
    cancelled.
    """

    def __init__(self, children, *, loop):
        assert loop is not None
        super().__init__(loop=loop)
        self._children = children
        self._cancel_requested = False

    def cancel(self, msg=None):
        if self.done():
            return False
        ret = False
        for child in self._children:
            if child.cancel(msg=msg):
                ret = True
        if ret:
            # If any child tasks were actually cancelled, we should
            # propagate the cancellation request regardless of
            # *return_exceptions* argument.  See issue 32684.
            self._cancel_requested = True
        return ret


def gather(*coros_or_futures, return_exceptions=False):
    """Return a future aggregating results from the given coroutines/futures.

    Coroutines will be wrapped in a future and scheduled in the event
    loop. They will not necessarily be scheduled in the same order as
    passed in.

    All futures must share the same event loop.  If all the tasks are
    done successfully, the returned future's result is the list of
    results (in the order of the original sequence, not necessarily
    the order of results arrival).  If *return_exceptions* is True,
    exceptions in the tasks are treated the same as successful
    results, and gathered in the result list; otherwise, the first
    raised exception will be immediately propagated to the returned
    future.

    Cancellation: if the outer Future is cancelled, all children (that
    have not completed yet) are also cancelled.  If any child is
    cancelled, this is treated as if it raised CancelledError --
    the outer Future is *not* cancelled in this case.  (This is to
    prevent the cancellation of one child to cause other children to
    be cancelled.)

    If *return_exceptions* is False, cancelling gather() after it
    has been marked done won't cancel any submitted awaitables.
    For instance, gather can be marked done after propagating an
    exception to the caller, therefore, calling ``gather.cancel()``
    after catching an exception (raised by one of the awaitables) from
    gather won't cancel any other awaitables.

    @gaojian:
    返回给定 coroutines/futures 的结果。

    协程将被包装在 future 中并在event loop中被调度执行。它们不一定按照传入的顺序被调度执行。

    所有 future对象 必须共享同一个event loop。  如果所有的任务都成功完成，将返回所有future对象的结果列表（按原顺序排列）。
    如果 *return_exceptions*为 True，任务中的异常被视为与成功相同的结果，并汇总到结果列表中；
    如果 *return_exceptions*为 False，第一个引发的异常将立即传播到返回的future对象。

    取消：如果外部 Future 被取消，所有子项（尚未完成的）也被取消。  
    如果有哪个孩子是已取消，这被视为引发了 CancelledError -- 在这种情况下，外部 Future 不会被取消。  （这是为了防止取消一个孩子导致其他孩子被取消。）
    
    如果 *return_exceptions*为 False，则取消其后的 Gather()已标记为完成不会取消任何提交的等待项。
    例如，可以在传播后将收集标记为完成
    调用者的异常，因此，调用“gather.cancel()”
    捕获异常（由其中一个可等待对象引发）后Gather 不会取消任何其他等待项。
    """
    if not coros_or_futures:
        loop = events._get_event_loop()
        outer = loop.create_future()
        outer.set_result([])
        return outer

    def _done_callback(fut):
        nonlocal nfinished
        nfinished += 1

        if outer is None or outer.done():
            if not fut.cancelled():
                # Mark exception retrieved.
                fut.exception()
            return

        if not return_exceptions:
            if fut.cancelled():
                # Check if 'fut' is cancelled first, as
                # 'fut.exception()' will *raise* a CancelledError
                # instead of returning it.
                exc = fut._make_cancelled_error()
                outer.set_exception(exc)
                return
            else:
                exc = fut.exception()
                if exc is not None:
                    outer.set_exception(exc)
                    return

        if nfinished == nfuts:
            # All futures are done; create a list of results
            # and set it to the 'outer' future.
            results = []

            for fut in children:
                if fut.cancelled():
                    # Check if 'fut' is cancelled first, as 'fut.exception()'
                    # will *raise* a CancelledError instead of returning it.
                    # Also, since we're adding the exception return value
                    # to 'results' instead of raising it, don't bother
                    # setting __context__.  This also lets us preserve
                    # calling '_make_cancelled_error()' at most once.
                    res = exceptions.CancelledError(
                        '' if fut._cancel_message is None else
                        fut._cancel_message)
                else:
                    res = fut.exception()
                    if res is None:
                        res = fut.result()
                results.append(res)

            if outer._cancel_requested:
                # If gather is being cancelled we must propagate the
                # cancellation regardless of *return_exceptions* argument.
                # See issue 32684.
                exc = fut._make_cancelled_error()
                outer.set_exception(exc)
            else:
                outer.set_result(results)

    arg_to_fut = {}

    # 所有的任务
    children = []

    # 任务数量
    nfuts = 0

    # 已完成任务数量
    nfinished = 0
    loop = None
    outer = None  # bpo-46672
    for arg in coros_or_futures:
        if arg not in arg_to_fut:
            fut = _ensure_future(arg, loop=loop)
            if loop is None:
                loop = futures._get_loop(fut)
            if fut is not arg:
                # 'arg' was not a Future, therefore, 'fut' is a new
                # Future created specifically for 'arg'.  Since the caller
                # can't control it, disable the "destroy pending task"
                # warning.

                # @gaojian:
                # “arg” 不是一个 Future对象，“fut” 是一个新的专门为“arg”创建的 Future对象。
                # 由于调用者无法控制它，因此禁用“销毁待处理任务”警告
                fut._log_destroy_pending = False

            nfuts += 1
            arg_to_fut[arg] = fut
            fut.add_done_callback(_done_callback)

        else:
            # There's a duplicate Future object in coros_or_futures.
            # @gaojian: 有重复的参数，直接从字典中获取Future对象
            fut = arg_to_fut[arg]

        children.append(fut)

    outer = _GatheringFuture(children, loop=loop)
    return outer


def shield(arg):
    """Wait for a future, shielding it from cancellation.

    The statement

        task = asyncio.create_task(something())
        res = await shield(task)

    is exactly equivalent to the statement

        res = await something()

    *except* that if the coroutine containing it is cancelled, the
    task running in something() is not cancelled.  From the POV of
    something(), the cancellation did not happen.  But its caller is
    still cancelled, so the yield-from expression still raises
    CancelledError.  Note: If something() is cancelled by other means
    this will still cancel shield().

    If you want to completely ignore cancellation (not recommended)
    you can combine shield() with a try/except clause, as follows:

        task = asyncio.create_task(something())
        try:
            res = await shield(task)
        except CancelledError:
            res = None

    Save a reference to tasks passed to this function, to avoid
    a task disappearing mid-execution. The event loop only keeps
    weak references to tasks. A task that isn't referenced elsewhere
    may get garbage collected at any time, even before it's done.

    @gaojian:
    shield() 函数用来保护一个future，使其不会被取消。
    例如：
        task = asyncio.create_task(something())
        res = await shield(task)
    与下面的语句等价：
        res = await something()
    但是，如果包含它的协程被取消，那么在 something() 中运行的任务不会被取消。
    从 something() 的角度来看，取消并没有发生。但是其调用者仍然被取消，因此 yield-from 表达式仍会引发 CancelledError。
    注意：如果 something() 被其他方式取消，这仍然会取消 shield()。
    如果要完全忽略取消（不建议），可以将 shield() 与 try/except 子句结合使用，如下所示：
        task = asyncio.create_task(something())
        try:
            res = await shield(task)
        except CancelledError:
            res = None
    保存对传递给此函数的任务的引用，以避免任务在执行过程中消失。
    事件循环只会对任务保留弱引用。没有其他引用的任务可能会在任何时候被垃圾回收，甚至在完成之前。
    """
    inner = _ensure_future(arg)
    if inner.done():
        # Shortcut.
        return inner
    loop = futures._get_loop(inner)
    outer = loop.create_future()

    def _inner_done_callback(inner):
        if outer.cancelled():
            if not inner.cancelled():
                # Mark inner's result as retrieved.
                inner.exception()
            return

        if inner.cancelled():
            outer.cancel()
        else:
            exc = inner.exception()
            if exc is not None:
                outer.set_exception(exc)
            else:
                outer.set_result(inner.result())


    def _outer_done_callback(outer):
        if not inner.done():
            inner.remove_done_callback(_inner_done_callback)

    inner.add_done_callback(_inner_done_callback)
    outer.add_done_callback(_outer_done_callback)
    return outer


def run_coroutine_threadsafe(coro, loop):
    """Submit a coroutine object to a given event loop.

    Return a concurrent.futures.Future to access the result.

    The coroutine will be wrapped in a future and scheduled in the event loop.
    The loop argument is the event loop in which the coroutine will be run.
    This function should be used when a coroutine needs to be started in a
    different thread. 

    @gaojian：这个函数是用来在不同的线程中启动一个协程的。

    在给定的事件循环中提交一个协程对象。
    返回一个 concurrent.futures.Future 以访问结果。

    该协程将被包装在一个 future 中，并在事件循环中调度。
    loop 参数是协程将在其中运行的事件循环。

    当需要在不同的线程中启动协程时，应使用此函数。
    """
    if not coroutines.iscoroutine(coro):
        raise TypeError('A coroutine object is required')
    future = concurrent.futures.Future()

    def callback():
        try:
            futures._chain_future(ensure_future(coro, loop=loop), future)
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            if future.set_running_or_notify_cancel():
                future.set_exception(exc)
            raise

    loop.call_soon_threadsafe(callback)
    return future


# WeakSet containing all alive tasks.
# @gaojian: 保存所有活着的任务
_all_tasks = weakref.WeakSet()

# Dictionary containing tasks that are currently active in
# all running event loops.  {EventLoop: Task}
# @gaojian: 保存每个loop当前执行的任务
_current_tasks = {}


def _register_task(task):
    """Register a new task in asyncio as executed by loop.  
    注册task
    """
    _all_tasks.add(task)


def _enter_task(loop, task):
    current_task = _current_tasks.get(loop)
    if current_task is not None:
        raise RuntimeError(f"Cannot enter into task {task!r} while another "
                           f"task {current_task!r} is being executed.")
    _current_tasks[loop] = task


def _leave_task(loop, task):
    current_task = _current_tasks.get(loop)
    if current_task is not task:
        raise RuntimeError(f"Leaving task {task!r} does not match "
                           f"the current task {current_task!r}.")
    del _current_tasks[loop]


def _unregister_task(task):
    """Unregister a task."""
    _all_tasks.discard(task)


_py_register_task = _register_task
_py_unregister_task = _unregister_task
_py_enter_task = _enter_task
_py_leave_task = _leave_task


try:
    from _asyncio import (_register_task, _unregister_task,
                          _enter_task, _leave_task,
                          _all_tasks, _current_tasks)
except ImportError:
    pass
else:
    _c_register_task = _register_task
    _c_unregister_task = _unregister_task
    _c_enter_task = _enter_task
    _c_leave_task = _leave_task
