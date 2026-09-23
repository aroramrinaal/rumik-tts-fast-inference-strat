import ctypes as C
import glob
import hashlib
from pathlib import Path

import torch


class _Conditional(C.Structure):
    _fields_ = [("handle", C.c_ulonglong), ("type", C.c_int),
                ("size", C.c_uint), ("graphs", C.POINTER(C.c_void_p))]


class _Payload(C.Union):
    _fields_ = [("reserved", C.c_longlong * 29), ("conditional", _Conditional)]


class _Node(C.Structure):

    _fields_ = [("type", C.c_int), ("reserved0", C.c_int * 3),
                ("payload", _Payload), ("reserved2", C.c_longlong)]


class _Kernel(C.Structure):

    _fields_ = [("func", C.c_void_p), ("gx", C.c_uint),
                ("gy", C.c_uint), ("gz", C.c_uint),
                ("bx", C.c_uint), ("by", C.c_uint), ("bz", C.c_uint),
                ("shared", C.c_uint), ("args", C.POINTER(C.c_void_p)),
                ("extra", C.POINTER(C.c_void_p))]


def _library(name):
    base = Path(torch.__file__).resolve().parent.parent
    candidates = sorted(glob.glob(str(base / "nvidia" / "*" / "lib" / name)))
    candidates += sorted(glob.glob(str(base / "torch" / "lib" / name)))
    if not candidates:
        raise RuntimeError(f"Bundled CUDA library not found: {name}")
    return C.CDLL(candidates[0])


def _bind(lib, name, args, result=C.c_int):
    function = getattr(lib, name)
    function.argtypes, function.restype = args, result
    return function


class ConditionalGraph:


    def __init__(self, body_graph, action, controls):
        if not action.is_cuda or action.dtype != torch.int32 or action.numel() != 1:
            raise ValueError("action must be one CUDA int32 value")
        if not controls.is_cuda or controls.dtype != torch.int64 or controls.numel() < 5:
            raise ValueError("controls must contain at least five CUDA int64 values")
        if not action.is_contiguous() or not controls.is_contiguous():
            raise ValueError("conditional control storage must be contiguous")
        self.body_graph, self.action, self.controls = body_graph, action, controls
        self.device = action.device
        self.runtime = _library("libcudart.so.12")
        self.nvrtc = _library("libnvrtc.so.12")
        self.driver = C.CDLL("libcuda.so.1")
        self.graph, self.executable, self.module = C.c_void_p(), C.c_void_p(), C.c_void_p()
        self._setup_bindings()
        version = C.c_int()
        self._check(self.runtime.cudaRuntimeGetVersion(C.byref(version)), "runtime version")
        if not 12040 <= version.value < 13000:
            raise RuntimeError(f"Expected frozen CUDA12.4+ runtime, found {version.value}")
        self.runtime_version = version.value
        self.function = self._compile_controller()
        driver_version = C.c_int()
        self._check(self.driver.cuDriverGetVersion(C.byref(driver_version)),
                    "driver version", driver=True)
        self.metadata = dict(runtime_version=self.runtime_version,
            driver_version=driver_version.value, control_nodes=2,
            body_ownership="retained PyTorch keep_graph capture pool; CUDA-cloned child",
            controller_cubin_sha256=self.controller_sha256,
            controller_reference="https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/tests/helpers/graph_kernels.py")
        try:
            self._build()
        except Exception:
            self.close()
            raise

    def _setup_bindings(self):
        p, pp, u, z = C.c_void_p, C.POINTER(C.c_void_p), C.c_uint, C.c_size_t
        r = self.runtime
        _bind(r, "cudaGetErrorString", [C.c_int], C.c_char_p)
        _bind(r, "cudaRuntimeGetVersion", [C.POINTER(C.c_int)])
        _bind(r, "cudaGraphCreate", [pp, u])
        _bind(r, "cudaGraphConditionalHandleCreate", [C.POINTER(C.c_ulonglong), p, u, u])
        _bind(r, "cudaGraphAddNode", [pp, p, pp, z, C.POINTER(_Node)])
        _bind(r, "cudaGraphAddChildGraphNode", [pp, p, pp, z, p])
        _bind(r, "cudaGraphInstantiateWithFlags", [pp, p, C.c_ulonglong])
        _bind(r, "cudaGraphLaunch", [p, p])
        _bind(r, "cudaGraphDestroy", [p])
        _bind(r, "cudaGraphExecDestroy", [p])
        d = self.driver
        _bind(d, "cuDriverGetVersion", [C.POINTER(C.c_int)])
        _bind(d, "cuModuleLoadData", [pp, p])
        _bind(d, "cuModuleGetFunction", [pp, p, C.c_char_p])
        _bind(d, "cuModuleUnload", [p])
        _bind(d, "cuGraphAddKernelNode", [pp, p, pp, z, C.POINTER(_Kernel)])
        _bind(d, "cuGetErrorString", [C.c_int, C.POINTER(C.c_char_p)])
        n = self.nvrtc
        _bind(n, "nvrtcCreateProgram", [pp, C.c_char_p, C.c_char_p, C.c_int, p, p])
        _bind(n, "nvrtcCompileProgram", [p, C.c_int, C.POINTER(C.c_char_p)])
        _bind(n, "nvrtcGetProgramLogSize", [p, C.POINTER(z)])
        _bind(n, "nvrtcGetProgramLog", [p, p])
        _bind(n, "nvrtcGetCUBINSize", [p, C.POINTER(z)])
        _bind(n, "nvrtcGetCUBIN", [p, p])
        _bind(n, "nvrtcDestroyProgram", [pp])

    def _check(self, code, operation, driver=False):
        if code:
            if driver:
                message = C.c_char_p()
                self.driver.cuGetErrorString(code, C.byref(message))
                detail = message.value
            else:
                detail = self.runtime.cudaGetErrorString(code)
            raise RuntimeError(f"{operation}: CUDA error {code}: {detail!r}")

    def _compile_controller(self):


        source = b'''
extern "C" __device__ __cudart_builtin__ void CUDARTAPI
cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);
extern "C" __global__ void update_condition(cudaGraphConditionalHandle handle,
        const int *action, const long long *controls) {
    cudaGraphSetConditional(handle, *action == 0 && controls[1] < controls[4]);
}
'''
        program = C.c_void_p()
        result = self.nvrtc.nvrtcCreateProgram(C.byref(program), source,
                                             b"conditional_controller.cu", 0, None, None)
        if result:
            raise RuntimeError(f"nvrtcCreateProgram failed: {result}")
        try:
            major, minor = torch.cuda.get_device_capability(self.device)
            options = (C.c_char_p * 2)(f"--gpu-architecture=sm_{major}{minor}".encode(),
                                       b"--std=c++17")
            result = self.nvrtc.nvrtcCompileProgram(program, len(options), options)
            size = C.c_size_t()
            self.nvrtc.nvrtcGetProgramLogSize(program, C.byref(size))
            log = C.create_string_buffer(max(1, size.value))
            self.nvrtc.nvrtcGetProgramLog(program, log)
            if result:
                raise RuntimeError(f"NVRTC conditional compile failed ({result}): {log.value.decode()}")
            result = self.nvrtc.nvrtcGetCUBINSize(program, C.byref(size))
            if result:
                raise RuntimeError(f"nvrtcGetCUBINSize failed: {result}")
            binary = C.create_string_buffer(size.value)
            result = self.nvrtc.nvrtcGetCUBIN(program, binary)
            if result:
                raise RuntimeError(f"nvrtcGetCUBIN failed: {result}")
            self.controller_sha256 = hashlib.sha256(binary.raw).hexdigest()
            self._check(self.driver.cuModuleLoadData(C.byref(self.module), binary),
                        "controller module load", driver=True)
            function = C.c_void_p()
            self._check(self.driver.cuModuleGetFunction(C.byref(function), self.module,
                        b"update_condition"), "controller lookup", driver=True)
            return function
        finally:
            self.nvrtc.nvrtcDestroyProgram(C.byref(program))

    def _controller_node(self, graph, dependency=None):
        node = C.c_void_p()
        values = (self.handle, C.c_void_p(self.action.data_ptr()),
                  C.c_void_p(self.controls.data_ptr()))
        arguments = (C.c_void_p * 3)(*[C.addressof(value) for value in values])
        params = _Kernel(self.function, 1, 1, 1, 1, 1, 1, 0, arguments, None)
        dependencies = (C.c_void_p * 1)(dependency.value) if dependency is not None else None
        self._check(self.driver.cuGraphAddKernelNode(C.byref(node), graph, dependencies,
                        int(dependency is not None), C.byref(params)),
                    "add condition controller", driver=True)
        return node

    def _build(self):
        self._check(self.runtime.cudaGraphCreate(C.byref(self.graph), 0), "create graph")
        self.handle = C.c_ulonglong()
        self._check(self.runtime.cudaGraphConditionalHandleCreate(C.byref(self.handle),
                    self.graph, 1, 1), "create conditional handle")

        initial = self._controller_node(self.graph)
        params = _Node()
        params.type = 13
        params.payload.conditional.handle = self.handle.value
        params.payload.conditional.type = 1
        params.payload.conditional.size = 1
        node = C.c_void_p()
        dependencies = (C.c_void_p * 1)(initial.value)
        self._check(self.runtime.cudaGraphAddNode(C.byref(node), self.graph,
                    dependencies, 1, C.byref(params)), "add WHILE node")
        body = C.c_void_p(params.payload.conditional.graphs[0])
        child = C.c_void_p()
        raw = C.c_void_p(self.body_graph.raw_cuda_graph())
        self._check(self.runtime.cudaGraphAddChildGraphNode(C.byref(child), body,
                    None, 0, raw), "add captured step child graph")
        self._controller_node(body, child)
        self._check(self.runtime.cudaGraphInstantiateWithFlags(C.byref(self.executable),
                    self.graph, 0), "instantiate conditional graph")

    def replay(self):
        stream = C.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
        self._check(self.runtime.cudaGraphLaunch(self.executable, stream), "launch conditional graph")

    def close(self):

        if self.executable.value:
            self.runtime.cudaGraphExecDestroy(self.executable)
            self.executable = C.c_void_p()
        if self.graph.value:
            self.runtime.cudaGraphDestroy(self.graph)
            self.graph = C.c_void_p()
        if self.module.value:
            self.driver.cuModuleUnload(self.module)
            self.module = C.c_void_p()
