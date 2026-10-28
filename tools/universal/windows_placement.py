"""Select physical Windows CPU cores before starting the CUDA worker."""
import ctypes
from ctypes import wintypes
import struct


def physical_cores(data):
    """Decode RelationProcessorCore records, retaining processor-group IDs."""
    records = []
    offset = 0
    while offset < len(data):
        relation, size = struct.unpack_from('<II', data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError('Invalid Windows CPU topology record')
        if relation == 0:
            count = struct.unpack_from('<H', data, offset + 30)[0]
            if count < 1 or size < 32 + 16 * count:
                raise ValueError('Invalid Windows processor group record')
            groups = [struct.unpack_from('<QH', data, offset + 32 + 16 * i)
                      for i in range(count)]
            records.append(groups)
        offset += size
    return records


def placement(count, requested_node=None):
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise ValueError('The Windows runtime requires a 64-bit interpreter')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.GetCurrentProcess()
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.GetActiveProcessorGroupCount.restype = wintypes.WORD
    if kernel.GetActiveProcessorGroupCount() != 1:
        raise ValueError('Multiple Windows processor groups require a qualified group-aware preset')
    highest = wintypes.ULONG()
    if not kernel.GetNumaHighestNodeNumber(ctypes.byref(highest)):
        raise ctypes.WinError(ctypes.get_last_error())
    if requested_node is None and highest.value != 0:
        raise ValueError('Multiple NUMA nodes detected; select the GPU-local node with --numa-node')
    node = 0 if requested_node is None else requested_node

    class GroupAffinity(ctypes.Structure):
        _fields_ = [('Mask', ctypes.c_size_t), ('Group', wintypes.WORD),
                    ('Reserved', wintypes.WORD * 3)]

    affinity = GroupAffinity()
    kernel.GetNumaNodeProcessorMaskEx.argtypes = [wintypes.USHORT, ctypes.POINTER(GroupAffinity)]
    if not 0 <= node <= highest.value or not kernel.GetNumaNodeProcessorMaskEx(node, ctypes.byref(affinity)):
        raise ValueError('Selected NUMA node has no available processors')
    allowed = process_mask.value & affinity.Mask
    size = wintypes.DWORD()
    kernel.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    if ctypes.get_last_error() != 122:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    selected = []
    for groups in physical_cores(buffer.raw[:size.value]):
        mask = next((mask for mask, group in groups if group == 0), 0) & allowed
        if mask:
            selected.append((mask & -mask).bit_length() - 1)
        if len(selected) == count:
            return node, selected
    raise ValueError(f'The selected preset requires {count} available physical CPU cores')


def apply_affinity(cpus):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), sum(1 << cpu for cpu in cpus)):
        raise ctypes.WinError(ctypes.get_last_error())
