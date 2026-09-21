"""Per-call numerical budgets inside the cancellable Linux planning process."""
import signal
import threading


def bounded_solver_call(function, seconds, *args, **kwargs):
    if seconds is None:
        return function(*args,**kwargs)
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('bounded_solver_requires_planning_process_main_thread')
    if not 0 < seconds <= 60:
        raise ValueError('invalid solver budget')
    def expired(_signum,_frame):
        raise TimeoutError('solver_call_timeout_{}s'.format(seconds))
    previous=signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        return function(*args,**kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
