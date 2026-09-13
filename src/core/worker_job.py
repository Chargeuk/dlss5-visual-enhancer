"""Windows job ownership: worker descendants die when their supervisor exits."""
import ctypes
import os
from ctypes import wintypes


def own_process(process):
    if os.name != 'nt': return None
    class Basic(ctypes.Structure):
        _fields_ = [('user_time',ctypes.c_int64),('job_time',ctypes.c_int64),
            ('flags',wintypes.DWORD),('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),
            ('process_limit',wintypes.DWORD),('affinity',ctypes.c_size_t),
            ('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]
    class Limits(ctypes.Structure):
        _fields_ = [('basic',Basic),('io',ctypes.c_uint64*6),
            ('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),
            ('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p,wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE,wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateJobObjectW(None,None)
    if not handle: raise ctypes.WinError(ctypes.get_last_error())
    limits = Limits(); limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(handle,9,ctypes.byref(limits),ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(handle,int(process._handle)):
        error = ctypes.get_last_error(); kernel.CloseHandle(handle)
        raise ctypes.WinError(error)
    return lambda: kernel.CloseHandle(handle)
