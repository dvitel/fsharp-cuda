module Cuda.DriverAPI
open System
open System.Runtime.InteropServices

type CUresult = 
    | CUDA_SUCCESS = 0
    | CUDA_ERROR_INVALID_VALUE = 1
    | CUDA_ERROR_OUT_OF_MEMORY = 2
    | CUDA_ERROR_NOT_INITIALIZED = 3
    | CUDA_ERROR_DEINITIALIZED = 4
    | CUDA_ERROR_PROFILER_DISABLED = 5
    | CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6
    | CUDA_ERROR_PROFILER_ALREADY_STARTED = 7
    | CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8
    | CUDA_ERROR_NO_DEVICE = 100
    | CUDA_ERROR_INVALID_DEVICE = 101
    | CUDA_ERROR_INVALID_IMAGE = 200
    | CUDA_ERROR_INVALID_CONTEXT = 201
    | CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202
    | CUDA_ERROR_MAP_FAILED = 205
    | CUDA_ERROR_UNMAP_FAILED = 206
    | CUDA_ERROR_ARRAY_IS_MAPPED = 207
    | CUDA_ERROR_ALREADY_MAPPED = 208
    | CUDA_ERROR_NO_BINARY_FOR_GPU = 209
    | CUDA_ERROR_ALREADY_ACQUIRED = 210
    | CUDA_ERROR_NOT_MAPPED = 211
    | CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212
    | CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213
    | CUDA_ERROR_ECC_UNCORRECTABLE = 214
    | CUDA_ERROR_UNSUPPORTED_LIMIT = 215
    | CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216
    | CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
    | CUDA_ERROR_INVALID_SOURCE = 300
    | CUDA_ERROR_FILE_NOT_FOUND = 301
    | CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
    | CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303
    | CUDA_ERROR_OPERATING_SYSTEM = 304
    | CUDA_ERROR_INVALID_HANDLE = 400
    | CUDA_ERROR_NOT_FOUND = 500
    | CUDA_ERROR_NOT_READY = 600
    | CUDA_ERROR_LAUNCH_FAILED = 700
    | CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
    | CUDA_ERROR_LAUNCH_TIMEOUT = 702
    | CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
    | CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
    | CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
    | CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708
    | CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
    | CUDA_ERROR_ASSERT = 710
    | CUDA_ERROR_TOO_MANY_PEERS = 711
    | CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
    | CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
    | CUDA_ERROR_NOT_PERMITTED = 800
    | CUDA_ERROR_NOT_SUPPORTED = 801
    | CUDA_ERROR_UNKNOWN = 999

[<DllImport("nvcuda.dll")>]
extern CUresult cuGetErrorName(CUresult error, IntPtr& pStr) 

[<DllImport("nvcuda.dll")>]
extern CUresult cuGetErrorString(CUresult error, IntPtr& pStr) 

type CUdevice = int

type CUdevice_attribute = 
    | CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
        // Maximum number of threads per block

    | CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2
        // Maximum block dimension X

    | CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3
        // Maximum block dimension Y

    | CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4
        // Maximum block dimension Z

    | CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5
        // Maximum grid dimension X

    | CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6
        //Maximum grid dimension Y

    | CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7
        //Maximum grid dimension Z

    | CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
        //Maximum shared memory available per block in bytes

    | CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8
        //Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK

    | CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9
        //Memory available on device for __constant__ variables in a CUDA C kernel in bytes

    | CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
        //Warp size in threads

    | CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11
        //Maximum pitch in bytes allowed by memory copies

    | CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12
        //Maximum number of 32-bit registers available per block

    | CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12
        //Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK

    | CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
        //Peak clock frequency in kilohertz

    | CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14
        //Alignment requirement for textures

    | CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15
        //Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT.

    | CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
        //Number of multiprocessors on device

    | CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17
        //Specifies whether there is a run time limit on kernels

    | CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
        //Device is integrated with host memory

    | CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19
        //Device can map host memory into CUDA address space

    | CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20
        //Compute mode (See CUcomputemode for details)

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21
        //Maximum 1D texture width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22
        //Maximum 2D texture width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23
        //Maximum 2D texture height

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24
        //Maximum 3D texture width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25
        //Maximum 3D texture height

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26
        //Maximum 3D texture depth

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27
        //Maximum 2D layered texture width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28
        //Maximum 2D layered texture height

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29
        //Maximum layers in a 2D layered texture

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27
        //Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28
        //Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29
        //Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS

    | CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30
        //Alignment requirement for surfaces

    | CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31
        //Device can possibly execute multiple kernels concurrently

    | CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32
        //Device has ECC support enabled

    | CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
        //PCI bus ID of the device

    | CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34
        //PCI device ID of the device

    | CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35
        //Device is using TCC driver model

    | CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
        //Peak memory clock frequency in kilohertz

    | CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
        //Global memory bus width in bits

    | CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
        //Size of L2 cache in bytes

    | CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
        //Maximum resident threads per multiprocessor

    | CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40
        //Number of asynchronous engines

    | CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41
        //Device shares a unified address space with the host

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42
        //Maximum 1D layered texture width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43
        //Maximum layers in a 1D layered texture

    | CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44
        //Deprecated, do not use.

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45
        //Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46
        //Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47
        //Alternate maximum 3D texture width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48
        //Alternate maximum 3D texture height

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49
        //Alternate maximum 3D texture depth

    | CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50
        //PCI domain ID of the device

    | CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51
        //Pitch alignment requirement for textures

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52
        //Maximum cubemap texture width/height

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53
        //Maximum cubemap layered texture width/height

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54
        //Maximum layers in a cubemap layered texture

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55
        //Maximum 1D surface width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56
        //Maximum 2D surface width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57
        //Maximum 2D surface height

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58
        //Maximum 3D surface width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59
        //Maximum 3D surface height

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60
        //Maximum 3D surface depth

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61
        //Maximum 1D layered surface width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62
        //Maximum layers in a 1D layered surface

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63
        //Maximum 2D layered surface width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64
        //Maximum 2D layered surface height

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65
        //Maximum layers in a 2D layered surface

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66
        //Maximum cubemap surface width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67
        //Maximum cubemap layered surface width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68
        //Maximum layers in a cubemap layered surface

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69
        //Maximum 1D linear texture width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70
        //Maximum 2D linear texture width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71
        //Maximum 2D linear texture height

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72
        //Maximum 2D linear texture pitch in bytes

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73
        //Maximum mipmapped 2D texture width

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74
        //Maximum mipmapped 2D texture height

    | CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
        //Major compute capability version number

    | CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
        //Minor compute capability version number

    | CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77
        //Maximum mipmapped 1D texture width

type ComputeMode = 
    | MultipleContextsPerDevice = 0
    | Exclusive = 1
    | NoContexts = 2
    | ExclusiveProcess = 3

type CUcontext = IntPtr
// initialization
[<DllImport("nvcuda.dll")>]
extern CUresult cuInit(UInt32 Flags)

// management
//Driver
[<DllImport("nvcuda.dll")>]
extern CUresult cuDriverGetVersion(int& driverVersion)

//Device
[<DllImport("nvcuda.dll")>]
extern CUresult cuDeviceGet(CUdevice& device, int ordinal)

[<DllImport("nvcuda.dll")>]
extern CUresult cuDeviceGetAttribute(int& pi, CUdevice_attribute attrib, CUdevice dev )

[<DllImport("nvcuda.dll")>]
extern CUresult cuDeviceGetCount(int& count)

[<DllImport("nvcuda.dll")>]
//extern CUresult cuDeviceGetName([<MarshalAs(UnmanagedType.LPArray)>]byte[] name, int len, CUdevice dev)
extern CUresult cuDeviceGetName([<MarshalAs(UnmanagedType.LPStr)>]System.Text.StringBuilder name, int len, CUdevice dev)

[<DllImport("nvcuda.dll")>]
extern CUresult cuDeviceTotalMem (UInt64& bytes, CUdevice dev)

//Context
[<Flags>]
type CUcontextFlags = 
    | CU_CTX_SCHED_AUTO = 0x00
    | CU_CTX_SCHED_SPIN = 0x01
    | CU_CTX_SCHED_YIELD = 0x02
    | CU_CTX_SCHED_BLOCKING_SYNC = 0x04
    | CU_CTX_MAP_HOST = 0x08
    | CU_CTX_LMEM_RESIZE_TO_MAX = 0x10

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxCreate(CUcontext& pctx, [<MarshalAs(UnmanagedType.U4)>]CUcontextFlags flags, CUdevice dev)

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxDestroy(CUcontext ctx)

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxGetApiVersion(CUcontext ctx, UInt32& version)

type CUfunc_cache = 
    | CU_FUNC_CACHE_PREFER_NONE = 0x00
        //no preference for shared memory or L1 (default)
    | CU_FUNC_CACHE_PREFER_SHARED = 0x01
        //prefer larger shared memory and smaller L1 cache
    | CU_FUNC_CACHE_PREFER_L1 = 0x02
        //prefer larger L1 cache and smaller shared memory
    | CU_FUNC_CACHE_PREFER_EQUAL = 0x03
        //prefer equal sized L1 cache and shared memory

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxGetCacheConfig (CUfunc_cache& pconfig)

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxGetCurrent (CUcontext& pctx)

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxGetDevice (CUdevice& device)

type CUlimit = 
    | CU_LIMIT_STACK_SIZE = 0x00
        //GPU thread stack size
    | CU_LIMIT_PRINTF_FIFO_SIZE = 0x01
        //GPU printf FIFO size
    | CU_LIMIT_MALLOC_HEAP_SIZE = 0x02
        //GPU malloc heap size
    | CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03
        //GPU device runtime launch synchronize depth
    | CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04
        //GPU device runtime pending launch count

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxGetLimit (int& pvalue, CUlimit limit)

type CUsharedconfig =
    | CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0x00
        //set default shared memory bank size
    | CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 0x01
        //set shared memory bank width to four bytes
    | CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02
        //set shared memory bank width to eight bytes    

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxGetSharedMemConfig (CUsharedconfig& pConfig)

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxGetStreamPriorityRange (int& least, int& greatest)

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxPopCurrent (CUcontext& pctx)

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxPushCurrent (CUcontext ctx)

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxSetCacheConfig (CUfunc_cache config)

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxSetCurrent (CUcontext ctx)

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxSetLimit (CUlimit limit, int value)

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxSetSharedMemConfig (CUsharedconfig config)

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxSynchronize()

type CUfunction = IntPtr
type CUmodule = IntPtr
type CUlinkState = IntPtr

type CUjitInputType = 
    | CU_JIT_INPUT_CUBIN = 0
    | CU_JIT_INPUT_PTX = 1
    | CU_JIT_INPUT_FATBINARY = 2
    | CU_JIT_INPUT_OBJECT = 3
    | CU_JIT_INPUT_LIBRARY = 4

type CUjitOption = 
    | CU_JIT_MAX_REGISTERS  = 0
    | CU_JIT_THREADS_PER_BLOCK = 1
    | CU_JIT_WALL_TIME = 2
    | CU_JIT_INFO_LOG_BUFFER = 3
    | CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
    | CU_JIT_ERROR_LOG_BUFFER = 5
    | CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
    | CU_JIT_OPTIMIZATION_LEVEL = 7
    | CU_JIT_TARGET_FROM_CUCONTEXT = 8
    | CU_JIT_TARGET = 9
    | CU_JIT_FALLBACK_STRATEGY = 10
    | CU_JIT_GENERATE_DEBUG_INFO = 11
    | CU_JIT_LOG_VERBOSE = 12
    | CU_JIT_GENERATE_LINE_INFO = 13
    | CU_JIT_CACHE_MODE = 14

type CUjitCacheMode = 
    | CU_JIT_CACHE_OPTION_NONE = 0
    | CU_JIT_CACHE_OPTION_CG = 1
    | CU_JIT_CACHE_OPTION_CA = 2

//Module
[<DllImport("nvcuda.dll")>]
extern CUresult cuLinkAddData(CUlinkState linkState, CUjitInputType targetType, IntPtr data, int size, [<MarshalAs(UnmanagedType.LPStr)>]string name, uint32 numOptions, [<MarshalAs(UnmanagedType.LPArray)>]CUjitOption[] options, [<MarshalAs(UnmanagedType.LPArray)>]IntPtr[] optionValues)

[<DllImport("nvcuda.dll")>]
extern CUresult cuLinkAddFile(CUlinkState linkState, CUjitInputType targetType, [<MarshalAs(UnmanagedType.LPStr)>]string path, uint32 numOptions, [<MarshalAs(UnmanagedType.LPArray)>]CUjitOption[] options, [<MarshalAs(UnmanagedType.LPArray)>]IntPtr[] optionValues)

[<DllImport("nvcuda.dll")>]
extern CUresult cuLinkComplete(CUlinkState linkState, IntPtr& cubin, int& size)

[<DllImport("nvcuda.dll")>]
extern CUresult cuLinkCreate(uint32 numOptions, CUjitOption[] options, IntPtr[] optionValues, CUlinkState& stateOut)

[<DllImport("nvcuda.dll")>]
extern CUresult cuLinkDestroy(CUlinkState state)

[<DllImport("nvcuda.dll")>]
extern CUresult cuModuleGetFunction (CUfunction& hfunc, CUmodule hmod, [<MarshalAs(UnmanagedType.LPStr)>]string name)

type CUdeviceptr = UInt32

[<DllImport("nvcuda.dll")>]
extern CUresult cuModuleGetGlobal (CUdeviceptr& dptr, int& bytes, CUmodule hmod, [<MarshalAs(UnmanagedType.LPStr)>]string name)

type CUsurfref = IntPtr

[<DllImport("nvcuda.dll")>]
extern CUresult cuModuleGetSurfRef (CUsurfref& pSurfRef, CUmodule hmod, [<MarshalAs(UnmanagedType.LPStr)>]string name)

type CUtexref = IntPtr

[<DllImport("nvcuda.dll")>]
extern CUresult cuModuleGetTexRef (CUtexref& pTexRef, CUmodule hmod, [<MarshalAs(UnmanagedType.LPStr)>]string name)

[<DllImport("nvcuda.dll")>]
extern CUresult cuModuleLoad (CUmodule& cuModule, [<MarshalAs(UnmanagedType.LPStr)>]string fname)

[<DllImport("nvcuda.dll")>]
extern CUresult cuModuleLoadData (CUmodule& cuModule, IntPtr image)

[<DllImport("nvcuda.dll")>]
extern CUresult cuModuleLoadDataEx (CUmodule& cuModule, IntPtr image, UInt32 numOptions, [<MarshalAs(UnmanagedType.LPArray)>]CUjitOption[] options, [<MarshalAs(UnmanagedType.LPArray)>]IntPtr[] optionValues)

[<DllImport("nvcuda.dll")>]
extern CUresult cuModuleLoadFatBinary (CUmodule& cuModule, IntPtr fatCubin)

[<DllImport("nvcuda.dll")>]
extern CUresult cuModuleUnload (CUmodule hmod)

type CUarray = IntPtr

//^(\w.*)$
type CUarray_format =
    | CU_AD_FORMAT_UNSIGNED_INT8 = 0x01
        // Unsigned 8-bit integers

    | CU_AD_FORMAT_UNSIGNED_INT16 = 0x02
        // Unsigned 16-bit integers

    | CU_AD_FORMAT_UNSIGNED_INT32 = 0x03
        // Unsigned 32-bit integers

    | CU_AD_FORMAT_SIGNED_INT8 = 0x08
        // Signed 8-bit integers

    | CU_AD_FORMAT_SIGNED_INT16 = 0x09
        // Signed 16-bit integers

    | CU_AD_FORMAT_SIGNED_INT32 = 0x0a
        // Signed 32-bit integers

    | CU_AD_FORMAT_HALF = 0x10
        // 16-bit floating point

    | CU_AD_FORMAT_FLOAT = 0x20
        // 32-bit floating point

[<Flags>]
type CUDA_ARRAY3D_DESCRIPTOR_FLAGS =
    | CUDA_ARRAY3D_NONE = 0x00
    | CUDA_ARRAY3D_LAYERED = 0x01
        //to enable creation of layered CUDA arrays. If this flag is set, Depth specifies the number of layers, not the depth of a 3D array.
    | CUDA_ARRAY3D_SURFACE_LDST = 0x02
        //to enable surface references to be bound to the CUDA array. If this flag is not set, cuSurfRefSetArray will fail when attempting to bind the CUDA array to a surface reference.
    | CUDA_ARRAY3D_CUBEMAP = 0x04
        //to enable creation of cubemaps. If this flag is set, Width must be equal to Height, and Depth must be six. If the CUDA_ARRAY3D_LAYERED flag is also set, then Depth must be a multiple of six.
    | CUDA_ARRAY3D_TEXTURE_GATHER = 0x08
        //to indicate that the CUDA array will be used for texture gather. Texture gather can only be performed on 2D CUDA arrays.

#nowarn "9"

[<Struct>]
[<StructLayout(LayoutKind.Sequential)>]
type CUDA_ARRAY3D_DESCRIPTOR =
    val mutable Width: UInt32
    val mutable Height: UInt32
    val mutable Depth: UInt32
    val mutable Format: CUarray_format
    val mutable NumChannels: int
    [<MarshalAs(UnmanagedType.U4)>]
    val mutable Flags: CUDA_ARRAY3D_DESCRIPTOR_FLAGS

//Memory
[<DllImport("nvcuda.dll")>]
extern CUresult cuArray3DCreate (CUarray& pHandle, CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray)

[<DllImport("nvcuda.dll")>]
extern CUresult cuArray3DGetDescriptor (CUDA_ARRAY3D_DESCRIPTOR& pArrayDescriptor, CUarray hArray)

[<Struct>]
[<StructLayout(LayoutKind.Sequential)>]
type CUDA_ARRAY_DESCRIPTOR =
    val mutable Width: UInt32
    val mutable Height: UInt32
    val mutable Format: CUarray_format
    val mutable NumChannels: UInt32

[<DllImport("nvcuda.dll")>]
extern CUresult cuArrayCreate (CUarray& pHandle, CUDA_ARRAY_DESCRIPTOR* pAllocateArray)

[<DllImport("nvcuda.dll")>]
extern CUresult cuArrayDestroy (CUarray hArray)

[<DllImport("nvcuda.dll")>]
extern CUresult cuArrayGetDescriptor (CUDA_ARRAY_DESCRIPTOR& pArrayDescriptor, CUarray hArray)

[<DllImport("nvcuda.dll")>]
extern CUresult cuDeviceGetByPCIBusId (CUdevice& dev, [<MarshalAs(UnmanagedType.LPStr)>]string pciBusId)

[<DllImport("nvcuda.dll")>]
extern CUresult cuDeviceGetPCIBusId([<MarshalAs(UnmanagedType.LPStr)>]System.Text.StringBuilder pciBusId, int len, CUdevice dev)

[<DllImport("nvcuda.dll")>]
extern CUresult cuIpcCloseMemHandle (CUdeviceptr dptr)

type CUipcEventHandle = IntPtr
type CUevent = IntPtr

[<DllImport("nvcuda.dll")>]
extern CUresult cuIpcGetEventHandle (CUipcEventHandle& pHandle, CUevent event)

type CUipcMemHandle = IntPtr

[<DllImport("nvcuda.dll")>]
extern CUresult cuIpcGetMemHandle (CUipcMemHandle& pHandle, CUdeviceptr dptr)

[<DllImport("nvcuda.dll")>]
extern CUresult cuIpcOpenEventHandle (CUevent& phEvent, CUipcEventHandle handle)

[<Flags>]
type CU_IPC_MEM_FLAGS = | CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1

[<DllImport("nvcuda.dll")>]
extern CUresult cuIpcOpenMemHandle(CUdeviceptr& pdptr, CUipcMemHandle handle, [<MarshalAs(UnmanagedType.U4)>]CU_IPC_MEM_FLAGS Flags)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemAlloc(CUdeviceptr& dptr, UInt32 bytesize)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemAllocHost(IntPtr& pp, UInt32 bytesize)

type CUmemAttach_flags = 
    | CU_MEM_ATTACH_GLOBAL = 0x1
    | CU_MEM_ATTACH_HOST = 0x2
    | CU_MEM_ATTACH_SINGLE = 0x4

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemAllocManaged(CUdeviceptr& dptr, uint32 bytesize, [<MarshalAs(UnmanagedType.U4)>]CUmemAttach_flags flags)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemAllocPitch (CUdeviceptr& dptr, UInt32& pPitch, UInt32 WidthInBytes, UInt32 Height, UInt32 ElementSizeBytes)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemFree (CUdeviceptr dptr)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemFreeHost (IntPtr p)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemGetAddressRange (CUdeviceptr& pbase, UInt32& psize, CUdeviceptr dptr)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemGetInfo (UInt32& free, UInt32& total)

type CU_MEMHOSTALLOC_FLAGS = 
    | CU_MEMHOSTALLOC_PORTABLE = 0x01
        //The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.
    | CU_MEMHOSTALLOC_DEVICEMAP = 0x02
        // Maps the allocation into the CUDA address space. The device pointer to the memory may be obtained by calling cuMemHostGetDevicePointer(). This feature is available only on GPUs with compute capability greater than or equal to 1.1.
    | CU_MEMHOSTALLOC_WRITECOMBINED = 0x04
        // Allocates the memory as write-combined (WC). WC memory can be transferred across the PCI Express bus more quickly on some system configurations, but cannot be read efficiently by most CPUs. WC memory is a good option for buffers that will be written by the CPU and read by the GPU via mapped pinned memory or host->device transfers.

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemHostAlloc(IntPtr& pp, UInt32 bytesize, [<MarshalAs(UnmanagedType.U4)>] CU_MEMHOSTALLOC_FLAGS Flags)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemHostGetDevicePointer (CUdeviceptr& pdptr, IntPtr p, UInt32 Flags)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemHostGetFlags ([<Out>][<MarshalAs(UnmanagedType.U4)>]CU_MEMHOSTALLOC_FLAGS& pFlags, IntPtr p)

type CU_MEMHOSTREGISTER_FLAGS = 
    | CU_MEMHOSTREGISTER_PORTABLE = 0x01
        //The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.
    | CU_MEMHOSTREGISTER_DEVICEMAP = 0x02
        // Maps the allocation into the CUDA address space. The device pointer to the memory may be obtained by calling cuMemHostGetDevicePointer(). This feature is available only on GPUs with compute capability greater than or equal to 1.1.

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemHostRegister (IntPtr p, UInt32 bytesize, [<MarshalAs(UnmanagedType.U4)>] CU_MEMHOSTREGISTER_FLAGS Flags)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemHostUnregister(IntPtr p)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpy (CUdeviceptr dst, CUdeviceptr src, UInt32 ByteCount)

type CUmemorytype = 
    | CU_MEMORYTYPE_HOST = 0x01
        //Host memory
    | CU_MEMORYTYPE_DEVICE = 0x02
        //Device memory
    | CU_MEMORYTYPE_ARRAY = 0x03
        //Array memory
    | CU_MEMORYTYPE_UNIFIED = 0x04
        //Unified device or host memory

[<Struct>]
[<StructLayout(LayoutKind.Sequential)>]
type CUDA_MEMCPY2D = 
    val mutable srcXInBytes: UInt32
    val mutable srcY: UInt32
    val mutable srcMemoryType: CUmemorytype
    val mutable srcHost: IntPtr
    val mutable srcDevice: CUdeviceptr
    val mutable srcArray: CUarray
    val mutable srcPitch: UInt32

    val mutable dstXInBytes: UInt32
    val mutable dstY: UInt32
    val mutable dstMemoryType: CUmemorytype
    val mutable dstHost: IntPtr
    val mutable dstDevice: CUdeviceptr
    val mutable dstArray: CUarray
    val mutable dstPitch: UInt32

    val mutable WidthInBytes: UInt32
    val mutable Height: UInt32

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpy2D (CUDA_MEMCPY2D* pCopy)

type CUstream = IntPtr

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpy2DAsync (CUDA_MEMCPY2D* pCopy, CUstream hStream)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpy2DUnaligned (CUDA_MEMCPY2D* pCopy)

[<Struct>]
[<StructLayout(LayoutKind.Sequential)>]
type CUDA_MEMCPY3D = 

    val mutable srcXInBytes: UInt32
    val mutable srcY: UInt32
    val mutable srcZ: UInt32
    val mutable srcLOD: UInt32

    val mutable srcMemoryType: CUmemorytype
    val mutable srcHost: IntPtr
    val mutable srcDevice: CUdeviceptr
    val mutable srcArray: CUarray
    val mutable srcPitch: UInt32
    val mutable srcHeight: UInt32

    val mutable dstXInBytes: UInt32
    val mutable dstY: UInt32
    val mutable dstZ: UInt32
    val mutable dstLOD: UInt32
    val mutable dstMemoryType: CUmemorytype
    val mutable dstHost: IntPtr
    val mutable dstDevice: CUdeviceptr
    val mutable dstArray: CUarray
    val mutable dstPitch: UInt32
    val mutable dstHeight: UInt32

    val mutable WidthInBytes: UInt32
    val mutable Height: UInt32
    val mutable Depth: UInt32

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpy3D (CUDA_MEMCPY3D* pCopy)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpy3DAsync (CUDA_MEMCPY3D* pCopy, CUstream hStream)

(* TODO
[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpy3DPeer (CUDA_MEMCPY3D_PEER* pCopy)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpy3DPeerAsync (CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream)
*)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyAsync (CUdeviceptr dst, CUdeviceptr src, UInt32 ByteCount, CUstream hStream)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyAtoA ( CUarray dstArray, UInt32 dstOffset, CUarray srcArray, UInt32 srcOffset, UInt32 ByteCount )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, UInt32 srcOffset, UInt32 ByteCount )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyAtoH ( IntPtr dstHost, CUarray srcArray, UInt32 srcOffset, UInt32 ByteCount )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyAtoHAsync ( IntPtr dstHost, CUarray srcArray, UInt32 srcOffset, UInt32 ByteCount, CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyDtoA ( CUarray dstArray, UInt32 dstOffset, CUdeviceptr srcDevice, UInt32 ByteCount)

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyDtoD ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, UInt32 ByteCount )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyDtoDAsync ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, UInt32 ByteCount, CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyDtoH ( IntPtr dstHost, CUdeviceptr srcDevice, UInt32 ByteCount )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyDtoHAsync ( IntPtr dstHost, CUdeviceptr srcDevice, UInt32 ByteCount, CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyHtoA ( CUarray dstArray, UInt32 dstOffset, IntPtr srcHost, UInt32 ByteCount )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyHtoAAsync ( CUarray dstArray, UInt32 dstOffset, IntPtr srcHost, UInt32 ByteCount, CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyHtoD ( CUdeviceptr dstDevice, IntPtr srcHost, UInt32 ByteCount )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyHtoDAsync ( CUdeviceptr dstDevice, IntPtr srcHost, UInt32 ByteCount, CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyPeer ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, UInt32 ByteCount )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemcpyPeerAsync ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, UInt32 ByteCount, CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemsetD16 ( CUdeviceptr dstDevice, UInt16 us, UInt32 N )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemsetD16Async ( CUdeviceptr dstDevice, UInt16 us, UInt32 N, CUstream stream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemsetD2D16 ( CUdeviceptr dstDevice, UInt32 dstPitch, UInt16 us, UInt32 Width, UInt32 Height )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemsetD2D16Async ( CUdeviceptr dstDevice, UInt32 dstPitch, UInt16 us, UInt32 Width, UInt32 Height, CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemsetD2D32 ( CUdeviceptr dstDevice, UInt32 dstPitch, UInt32 ui, UInt32 Width, UInt32 Height )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemsetD2D32Async ( CUdeviceptr dstDevice, UInt32 dstPitch, UInt32 ui, UInt32 Width, UInt32 Height, CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemsetD2D8 ( CUdeviceptr dstDevice, UInt32 dstPitch, byte uc, UInt32 Width, UInt32 Height )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemsetD2D8Async ( CUdeviceptr dstDevice, UInt32 dstPitch, byte uc, UInt32 Width, UInt32 Height, CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemsetD32 ( CUdeviceptr dstDevice, UInt32 ui, UInt32 N )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemsetD32Async ( CUdeviceptr dstDevice, UInt32 ui, UInt32 N, CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemsetD8 ( CUdeviceptr dstDevice, byte uc, UInt32 N )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMemsetD8Async ( CUdeviceptr dstDevice, byte uc, UInt32 N, CUstream hStream )

type CUmipmappedArray = IntPtr

[<DllImport("nvcuda.dll")>]
extern CUresult cuMipmappedArrayCreate ( CUmipmappedArray& pHandle, CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, UInt32 numMipmapLevels )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMipmappedArrayDestroy ( CUmipmappedArray hMipmappedArray )

[<DllImport("nvcuda.dll")>]
extern CUresult cuMipmappedArrayGetLevel ( CUarray& pLevelArray, CUmipmappedArray hMipmappedArray, UInt32 level )

type CUpointer_attribute =
    | CU_POINTER_ATTRIBUTE_CONTEXT = 1
        //The CUcontext on which a pointer was allocated or registered
    | CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
        //The CUmemorytype describing the physical location of a pointer
    | CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
        //The address at which a pointer's memory may be accessed on the device
    | CU_POINTER_ATTRIBUTE_HOST_POINTER = 4
        //The address at which a pointer's memory may be accessed on the host
    | CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5
        //A pair of tokens for use with the nv-p2p.h Linux kernel interface

// Unified adressing
[<DllImport("nvcuda.dll")>]
extern CUresult cuPointerGetAttribute ( IntPtr& data, CUpointer_attribute attribute, CUdeviceptr ptr )

//Stream
type CUstreamCallback = delegate of (CUstream*CUresult*IntPtr) -> unit

[<DllImport("nvcuda.dll")>]
extern CUresult cuStreamAddCallback ( CUstream hStream, CUstreamCallback callback, IntPtr userData, UInt32 flags )

[<Flags>]
type CU_STREAM_FLAGS =  
    | CU_STREAM_DEFAULT = 0x0
        //Default stream creation flag.
    | CU_STREAM_NON_BLOCKING = 0x01
        // Specifies that work running in the created stream may run concurrently with work in stream 0 (the NULL stream), and that the created stream should perform no implicit synchronization with stream 0.

[<DllImport("nvcuda.dll")>]
extern CUresult cuStreamAttachMemAsync( CUstream hStream, CUdeviceptr dptr, int length, [<MarshalAs(UnmanagedType.U4)>] CUmemAttach_flags flags)

[<DllImport("nvcuda.dll")>]
extern CUresult cuStreamCreate ( CUstream& phStream, [<MarshalAs(UnmanagedType.U4)>] CU_STREAM_FLAGS Flags )

[<DllImport("nvcuda.dll")>]
extern CUresult cuStreamCreateWithPriority  ( CUstream& phStream, [<MarshalAs(UnmanagedType.U4)>] CU_STREAM_FLAGS Flags, int priority )

[<DllImport("nvcuda.dll")>]
extern CUresult cuStreamGetFlags ( CUstream hStream, int& flags )

[<DllImport("nvcuda.dll")>]
extern CUresult cuStreamGetPriority ( CUstream hStream, int& priority )

[<DllImport("nvcuda.dll")>]
extern CUresult cuStreamDestroy ( CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuStreamQuery ( CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuStreamSynchronize ( CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuStreamWaitEvent ( CUstream hStream, CUevent hEvent, UInt32 Flags )

//Event
[<Flags>]
type CU_EVENT_FLAGS = 
    | CU_EVENT_DEFAULT = 0
        //Default event creation flag.
    | CU_EVENT_BLOCKING_SYNC = 1
        // Specifies that the created event should use blocking synchronization. A CPU thread that uses cuEventSynchronize() to wait on an event created with this flag will block until the event has actually been recorded.
    | CU_EVENT_DISABLE_TIMING = 2
        // Specifies that the created event does not need to record timing data. Events created with this flag specified and the CU_EVENT_BLOCKING_SYNC flag not specified will provide the best performance when used with cuStreamWaitEvent() and cuEventQuery().
    | CU_EVENT_INTERPROCESS = 4
        // Specifies that the created event may be used as an interprocess event by cuIpcGetEventHandle(). CU_EVENT_INTERPROCESS must be specified along with CU_EVENT_DISABLE_TIMING.

[<DllImport("nvcuda.dll")>]
extern CUresult cuEventCreate ( CUevent& phEvent, [<MarshalAs(UnmanagedType.U4)>] CU_EVENT_FLAGS Flags )

[<DllImport("nvcuda.dll")>]
extern CUresult cuEventDestroy ( CUevent hEvent )

[<DllImport("nvcuda.dll")>]
extern CUresult cuEventElapsedTime ( float& pMilliseconds, CUevent hStart, CUevent hEnd )

[<DllImport("nvcuda.dll")>]
extern CUresult cuEventQuery ( CUevent hEvent )

[<DllImport("nvcuda.dll")>]
extern CUresult cuEventRecord ( CUevent hEvent, CUstream hStream )

[<DllImport("nvcuda.dll")>]
extern CUresult cuEventSynchronize ( CUevent hEvent )

//Execution control

type CUfunction_attribute = 
    | CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
        //The maximum number of threads per block, beyond which a launch of the function would fail. This number depends on both the function and the device on which the function is currently loaded.
    | CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
        //The size in bytes of statically-allocated shared memory required by this function. This does not include dynamically-allocated shared memory requested by the user at runtime.
    | CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
        //The size in bytes of user-allocated constant memory required by this function.
    | CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
        //The size in bytes of local memory used by each thread of this function.
    | CU_FUNC_ATTRIBUTE_NUM_REGS = 4
        //The number of registers used by each thread of this function.
    | CU_FUNC_ATTRIBUTE_PTX_VERSION = 5
        //The PTX virtual architecture version for which the function was compiled. This value is the major PTX version * 10 + the minor PTX version, so a PTX version 1.3 function would return the value 13. Note that this may return the undefined value of 0 for cubins compiled prior to CUDA 3.0.
    | CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6
        //The binary architecture version for which the function was compiled. This value is the major binary version * 10 + the minor binary version, so a binary version 1.3 function would return the value 13. Note that this will return a value of 10 for legacy cubins that do not have a properly-encoded binary architecture version.
    | CU_FUNC_CACHE_MODE_CA = 7

[<DllImport("nvcuda.dll")>]
extern CUresult cuFuncGetAttribute ( int& pi, CUfunction_attribute attrib, CUfunction hfunc )

[<DllImport("nvcuda.dll")>]
extern CUresult cuFuncSetCacheConfig ( CUfunction hfunc, CUfunc_cache config )

[<DllImport("nvcuda.dll")>]
extern CUresult cuFuncSetSharedMemConfig ( CUfunction hfunc, CUsharedconfig config )

let CU_LAUNCH_PARAM_END = IntPtr.Zero
let CU_LAUNCH_PARAM_BUFFER_POINTER = IntPtr 0x01
let CU_LAUNCH_PARAM_BUFFER_SIZE = IntPtr 0x02

[<DllImport("nvcuda.dll")>]
extern CUresult cuLaunchKernel ( CUfunction f, UInt32 gridDimX, UInt32 gridDimY, UInt32 gridDimZ, UInt32 blockDimX, UInt32 blockDimY, UInt32 blockDimZ, UInt32 sharedMemBytes, CUstream hStream, [<MarshalAs(UnmanagedType.LPArray)>]IntPtr[] kernelParams, [<MarshalAs(UnmanagedType.LPArray)>]IntPtr[] extra )

//TODO: Texture reference management
//TODO: Surface reference management
//TODO: Texture object management
//TODO: Surface object management

// Peer memory access

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxDisablePeerAccess ( CUcontext peerContext )

[<DllImport("nvcuda.dll")>]
extern CUresult cuCtxEnablePeerAccess ( CUcontext peerContext, UInt32 Flags )

[<DllImport("nvcuda.dll")>]
extern CUresult cuDeviceCanAccessPeer ( int& canAccessPeer, CUdevice dev, CUdevice peerDev )

//TODO: Graphics Interoperability

//Profiler control 

type CU_PROFILER_OPTIONS =  
    | CU_OUT_KEY_VALUE_PAIR = 0x0
    | CU_OUT_CSV = 0x01

[<DllImport("nvcuda.dll")>]
extern CUresult cuProfilerInitialize ( [<MarshalAsAttribute(UnmanagedType.LPStr)>]string configFile, [<MarshalAsAttribute(UnmanagedType.LPStr)>]string outputFile, [<MarshalAsAttribute(UnmanagedType.U4)>]CU_PROFILER_OPTIONS outputMode )

[<DllImport("nvcuda.dll")>]
extern CUresult cuProfilerStart ()

[<DllImport("nvcuda.dll")>]
extern CUresult cuProfilerStop ()

//TODO: OpenGL Interoperability
//TODO: Direct3D 9 Interoperability
//TODO: Direct3D 10 Interoperability
//TODO: Direct3D 11 Interoperability
//VDPAU Interoperability

//utils
type IntPtr with
    static member Pinned(v) = 
        let handle = GCHandle.Alloc(v :> obj, GCHandleType.Pinned)
        handle.AddrOfPinnedObject(), handle.Free

//    static member InUnmanagedHeap(v) = 
//        let v = v :> obj
//        let ptr = Marshal.AllocHGlobal(Marshal.SizeOf)
//        Marshal.WriteInt32(ptr, v)            
//        ptr
//        let freeIntPtr = Marshal.FreeHGlobal
//        (*use pinned managed memory*)
//        let getIntPtr v = 
//            let handle = GCHandle.Alloc(v :> obj, GCHandleType.Pinned)
//            handle.AddrOfPinnedObject()
//        //let freeIntPtr = GCHandle.FromIntPtr >> fun h -> h.Free()    

let inline (++) a (b, c) = (a, b, c)