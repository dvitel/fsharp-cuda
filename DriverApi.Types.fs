namespace Cuda
open System
open Cuda.DriverAPI
open System.Runtime.CompilerServices
open System.Runtime.InteropServices    

[<AllowNullLiteral>]
type CudaException(cuFunc, code: CUresult) =     
    inherit Exception()
    member x.Code = code
    override x.Message = 
        let mutable msgPtr = IntPtr.Zero
        cuGetErrorString(CUresult.CUDA_ERROR_ASSERT, &msgPtr) |> ignore        
        Marshal.PtrToStringAnsi(msgPtr)
    member x.FunctionName = cuFunc
    override x.ToString() = 
        sprintf "%s %d [%s] %A" x.Message (int x.Code) x.FunctionName x.StackTrace

module Driver = 
    let initializationResult = cuInit(0u)
    if initializationResult <> CUresult.CUDA_SUCCESS then
        printfn "CUDA initialization failed: %A" initializationResult //TODO trace to file or event log
    let version =   
        lazy
        let mutable v = 0
        let result = cuDriverGetVersion(&v)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuDriverGetVersion", result)
        else v

type Device internal (device: CUdevice, index) = 
    static let mutable count: int option = None        
    static let getAt index = 
        let mutable device:CUdevice = 0
        let result = cuDeviceGet(&device, index)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuDeviceGet", result)
        else Device(device, index)
    let mutable name = ""
    let mutable totalMemory = 0UL
    let mutable attributes: Map<CUdevice_attribute, int> = Map.empty
    let getAttribute attribute = 
        let mutable value = 0
        let result = cuDeviceGetAttribute(&value, attribute, device)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuDeviceGetAttribute", result)
        else value 
    let getAttributeCached attribute = 
        match Map.tryFind attribute attributes with
        | None ->
            let result = getAttribute attribute
            attributes <- Map.add attribute result attributes
            result
        | Some attribute -> attribute
    static member All = 
        seq { for i in 0..Device.Count-1 -> getAt i }
    static member FromPCI(pciId: string) = 
        let mutable dev = CUdevice.MinValue
        let result = cuDeviceGetByPCIBusId(&dev, pciId)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuDeviceGetByPCIBusId", result)
        Device(dev, 0)
    static member Count = 
        match count with
        | Some count -> count
        | None ->
            let mutable c = 0
            let result = cuDeviceGetCount(&c)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuDeviceGetCount", result)
            else 
                count <- Some c
                c    
    member x.Index = index
    member x.Name =         
        match name with
        | "" ->
            let mutable nameBuilder = System.Text.StringBuilder() // Array.zeroCreate<byte> 1000
            let result = cuDeviceGetName(nameBuilder, 1000, device)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuDeviceGetName", result)
            else 
                name <- nameBuilder.ToString() //System.Text.Encoding.ASCII.GetString(name)
                name
        | _ -> name
    member x.TotalMemory = 
        match totalMemory with
        | 0UL ->
            let result = cuDeviceTotalMem(&totalMemory, device)
            if result <> CUresult.CUDA_SUCCESS then
                totalMemory <- 0UL
                raise <| CudaException("cuDeviceTotalMem", result)
        | _ -> ()
        totalMemory
    member x.WarpSize = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE
    member x.ThreadsPerBlock = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    member x.BlockSizes = 
        getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
        getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
        getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
    member x.GridSizes = 
        getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
        getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z        
    member x.SharedMemoryPerBlock = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK 
    member x.ConstantMemory = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY
    member x.MaxPitch = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PITCH
    member x.RegistersPerBlock = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
    member x.ClockRate = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE
    member x.TextureAlignment = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT
    member x.MultiprocessorCount = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
    member x.KernelTimeout = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT
    member x.IsIntergrated = (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED) = 1
    member x.CanMapMemory = (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY) = 1
    member x.IsMultipleContextsAllowed = (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE) = 0
    member x.IsOnlyOneContextPerThread = (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE) = 1
    member x.IsOnlyOneContextPerProcess = (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE) = 3
    member x.IsContextDisallowed = (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE) = 2
    member x.ComputeMode = (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE) |> enum<ComputeMode>
    member x.Texture1DWidth = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH
    member x.Texture2DSizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT)
    member x.Texture3DSizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH)
    member x.LayeredTexture2DSizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS)
    member x.Texture2DArraySizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES)
    member x.SurfaceAlignment = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT    
    member x.CanExecuteMultipleKernels = (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS) = 1
    member x.IsEccEnabled = (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ECC_ENABLED) = 1
    member x.PciBusId = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID    
    member x.IsTccDriverModel = (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TCC_DRIVER) = 1
    member x.MemoryClockRateKHz = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE    
    member x.GlobalMemoryBusBitWidth = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH    
    member x.L2CacheSize = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE    
    member x.ThreadsPerMultiprocessor = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR    
    member x.AsyncEnginesCount = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT    
    member x.IsUnifiedAddressingSupported = (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING) = 1
    member x.LayeredTexture1DSizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS)
    member x.GatherTexture2DSizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT)        
    member x.AlternateTexture3DSizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE)        
    member x.PciDomainId = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID
    member x.PitchTextureAlignment = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT
    member x.CubemapTextureWidth = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH
    member x.CubemapLayeredTextureSizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS)        
    member x.Surface1DWidth = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH
    member x.Surface2DSizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT)    
    member x.Surface3DSizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH)          
    member x.LayeredSurface1DWidth = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS)
    member x.LayeredSurface2DSizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS)          
    member x.CubemapSurfaceWidth = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH
    member x.CubemapLayeredSurfaceSizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS)
    member x.LinearTexture1DWidth = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH
    member x.LinearTexture2DSizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH)   
    member x.MipmappedTexture2DSizes = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT)             
    member x.ComputeCapability = 
        (getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
         getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)            
    member x.MipmappedTexture1DWidth = getAttributeCached CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH
    member x.Item with get(attribute:CUdevice_attribute) = getAttributeCached attribute
    member private x.CreateContext(flags, includeMappedPinnedAllocations, noReduceOnLocalMemoryResize) = 
        let mutable flags = flags
        flags <- 
            if includeMappedPinnedAllocations then 
                CUcontextFlags.CU_CTX_MAP_HOST ||| flags
            else flags
        flags <- 
            if noReduceOnLocalMemoryResize then 
                CUcontextFlags.CU_CTX_LMEM_RESIZE_TO_MAX ||| flags
            else flags
        let mutable context = CUcontext.Zero 
        let result = cuCtxCreate(&context, flags, device)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuCtxCreate", result)
        else new Context(context)
    member x.CreateContext(?includeMappedPinnedAllocations, ?noReduceOnLocalMemoryResize) = 
        x.CreateContext(CUcontextFlags.CU_CTX_SCHED_AUTO, 
            defaultArg includeMappedPinnedAllocations false,
            defaultArg noReduceOnLocalMemoryResize false)
    member x.CreateSpinContext(?includeMappedPinnedAllocations, ?noReduceOnLocalMemoryResize) = 
        x.CreateContext(CUcontextFlags.CU_CTX_SCHED_SPIN, 
            defaultArg includeMappedPinnedAllocations false,
            defaultArg noReduceOnLocalMemoryResize false)        
    member x.CreateYeildContext(?includeMappedPinnedAllocations, ?noReduceOnLocalMemoryResize) = 
        x.CreateContext(CUcontextFlags.CU_CTX_SCHED_YIELD, 
            defaultArg includeMappedPinnedAllocations false,
            defaultArg noReduceOnLocalMemoryResize false)
    member x.CreateBlockingContext(?includeMappedPinnedAllocations, ?noReduceOnLocalMemoryResize) = 
        x.CreateContext(CUcontextFlags.CU_CTX_SCHED_BLOCKING_SYNC, 
            defaultArg includeMappedPinnedAllocations false,
            defaultArg noReduceOnLocalMemoryResize false)
    member x.GetPCI() =
        let mutable pciId = System.Text.StringBuilder(1024)
        let result = cuDeviceGetPCIBusId(pciId, 1024, device)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuDeviceGetPCIBusId", result)
        pciId.ToString()
         
and Context internal (context: CUcontext) as x = 
    let mutable apiVersion = 0u
    let setLimit limit value = 
        let result = cuCtxSetLimit(limit, value)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuCtxSetLimit", result) 
        x   
    let getLimit limit =  
        let mutable l = 0
        let result = cuCtxGetLimit(&l, limit)
        if result <> CUresult.CUDA_SUCCESS then                
            raise <| CudaException("cuCtxGetLimit", result)          
        else l
    let setCachePreference pref = 
        let result = cuCtxSetCacheConfig(pref)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuCtxSetCacheConfig", result)
        x
    let isPrefered preference = 
        let mutable pref = CUfunc_cache.CU_FUNC_CACHE_PREFER_NONE
        let result = cuCtxGetCacheConfig(&pref)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuCtxGetCacheConfig", result)  
        else pref = preference
    interface IDisposable with
        override x.Dispose() = 
            cuCtxDestroy(context) |> ignore //silent fails
    member x.ApiVersion = 
        match apiVersion with
        | 0u -> 
            let result = cuCtxGetApiVersion(context, &apiVersion)
            if result <> CUresult.CUDA_SUCCESS then
                apiVersion <- 0u
                raise <| CudaException("cuCtxGetApiVersion", result)
        | _ -> ()
        apiVersion
    member x.GetDevice() = 
        let mutable device = CUdevice.MinValue
        let result = cuCtxGetDevice(&device)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuCtxGetDevice", result)    
        else Device(device, 0) 
    member x.SetThreadStackSize(newSize) = setLimit CUlimit.CU_LIMIT_STACK_SIZE newSize
    member x.SetPrintfQueueSize(newSize) = setLimit CUlimit.CU_LIMIT_PRINTF_FIFO_SIZE
    member x.SetMallocHeapSize(newSize) = setLimit CUlimit.CU_LIMIT_MALLOC_HEAP_SIZE  
    member x.SetDeviceRuntimeSynchronizationDepth(newDepth) = setLimit CUlimit.CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH
    member x.SetDeviceRuntimeLaunchNumber(newCount) = setLimit CUlimit.CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT                             
    member x.ThreadStackSize = getLimit CUlimit.CU_LIMIT_STACK_SIZE
    member x.PrintfQueueSize = getLimit CUlimit.CU_LIMIT_PRINTF_FIFO_SIZE
    member x.MallocHeapSize = getLimit CUlimit.CU_LIMIT_MALLOC_HEAP_SIZE
    member x.DeviceRuntimeSynchronizationDepth = getLimit CUlimit.CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH
    member x.DeviceRuntimeLaunchNumber = getLimit CUlimit.CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT
    member x.SharedMemoryBankByteSize 
        with get() = 
            let mutable bankSize = CUsharedconfig.CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE
            let result = cuCtxGetSharedMemConfig(&bankSize)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuCtxGetSharedMemConfig", result)
            else
                match bankSize with
                | CUsharedconfig.CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE -> 4
                | CUsharedconfig.CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE -> 8
                | _ -> 0
        and set(size) = 
            let size =
                match size with
                | 4 -> CUsharedconfig.CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE
                | 8 -> CUsharedconfig.CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE
                | _ -> CUsharedconfig.CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE
            let result = cuCtxSetSharedMemConfig(size)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuCtxSetSharedMemConfig", result)
    member x.StreamPriorityRange = 
        let mutable greatestPriority = 0
        let mutable leastPriority = 0
        let result = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuCtxGetStreamPriorityRange", result)
        else (greatestPriority, leastPriority)
    member x.PreferL1InsteadSharedMemory() = setCachePreference CUfunc_cache.CU_FUNC_CACHE_PREFER_L1
    member x.PreferSharedMemoryInsteadL1() = setCachePreference CUfunc_cache.CU_FUNC_CACHE_PREFER_SHARED
    member x.PreferSharedMemoryEqualL1() = setCachePreference CUfunc_cache.CU_FUNC_CACHE_PREFER_EQUAL
    member x.PreferDefaultCaching() = setCachePreference CUfunc_cache.CU_FUNC_CACHE_PREFER_NONE
    member x.IsL1Prefered = isPrefered CUfunc_cache.CU_FUNC_CACHE_PREFER_L1 
    member x.IsSharedMemoryPrefered = isPrefered CUfunc_cache.CU_FUNC_CACHE_PREFER_SHARED 
    member x.IsSharedMemoryEqualToL1 = isPrefered CUfunc_cache.CU_FUNC_CACHE_PREFER_EQUAL                       
    member private x.CuContext = context
    member x.Dispose() = (x :> IDisposable).Dispose()
    member x.Synchronize() = 
        let result = cuCtxSynchronize()
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuCtxSynchronize", result)
    static member Pop() = 
        let mutable context = CUcontext.Zero
        let result = cuCtxPopCurrent(&context)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuCtxPopCurrent", result)
        else new Context(context)
    static member Push(context: Context) =
        let result = cuCtxPushCurrent(context.CuContext)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuCtxPushCurrent", result)
    static member GetCurrent() = 
        let mutable context = CUcontext.Zero
        let result = cuCtxGetCurrent(&context)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuCtxGetCurrent", result)
        else new Context(context)
    static member SetCurrent(context: Context) = 
        let result = cuCtxSetCurrent(context.CuContext)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuCtxSetCurrent", result)
    member x.FreeAndTotalMemories = 
        let mutable free = 0u
        let mutable total = 0u
        let result = cuMemGetInfo(&free, &total)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuMemGetInfo", result)
        (free, total)
        
type JitTarget = 
    | _10 = 10
    | _11 = 11
    | _12 = 12
    | _13 = 13
    | _20 = 20
    | _21 = 21
    | _30 = 30
    | _32 = 32
    | _35 = 35
    | _50 = 50
    | _52 = 52
type JitFallbackStrategy = | PTX = 0 | BINARY = 1
type JitOptionsTarget = | ContextTarget | ComputeTarget of JitTarget | MinThreadsPerBlock of uint32
type JitOptionsOptimization = | _0 = 0 | _1 = 1 | _2 = 2 | _3 = 3 | _4 = 4
type JitOptionsCacheMode = DlcmNone = 0 | L1Disabled = 1 | L1Enabled = 2
type JitOptions = {
    maxRegistersPerThread: uint32 option
    optimizationLevel: JitOptionsOptimization
    target: JitOptionsTarget
    fallback: JitFallbackStrategy option
    generateDebugInfo: bool
    verboseLogging: bool
    generateLineNumberInfo: bool
    cacheMode: JitOptionsCacheMode option
    includeTimingInfo: bool
    collectLogs: bool
}
with 
    static member Default = {
        maxRegistersPerThread = None
        optimizationLevel = JitOptionsOptimization._4
        target = ContextTarget
        fallback = None
        generateDebugInfo = false
        verboseLogging = false
        generateLineNumberInfo = false
        cacheMode = None
        includeTimingInfo = false
        collectLogs = false
    }
    member x.ToArrays() = 
        [|
            if x.maxRegistersPerThread.IsSome then 
                yield CUjitOption.CU_JIT_MAX_REGISTERS ++ IntPtr.Pinned x.maxRegistersPerThread.Value
            if x.optimizationLevel <> JitOptionsOptimization._4 then 
                yield CUjitOption.CU_JIT_OPTIMIZATION_LEVEL ++ IntPtr.Pinned(int x.optimizationLevel)
            match x.target with
            | ContextTarget -> ()
            | ComputeTarget target -> 
                yield CUjitOption.CU_JIT_TARGET ++ IntPtr.Pinned(int target)
            | MinThreadsPerBlock i -> 
                yield CUjitOption.CU_JIT_THREADS_PER_BLOCK ++ IntPtr.Pinned i
            match x.fallback with
            | None -> ()
            | Some strategy -> 
                yield CUjitOption.CU_JIT_FALLBACK_STRATEGY ++ IntPtr.Pinned(int strategy)
            if x.generateDebugInfo then yield CUjitOption.CU_JIT_GENERATE_DEBUG_INFO ++ IntPtr.Pinned 1u
            if x.verboseLogging then yield CUjitOption.CU_JIT_LOG_VERBOSE ++ IntPtr.Pinned 1u
            if x.generateLineNumberInfo then yield CUjitOption.CU_JIT_GENERATE_LINE_INFO ++ IntPtr.Pinned 1u
            match x.cacheMode with
            | None -> ()
            | Some cacheMode -> 
                yield CUjitOption.CU_JIT_CACHE_MODE ++ IntPtr.Pinned(int cacheMode)
            if x.includeTimingInfo then 
                yield CUjitOption.CU_JIT_WALL_TIME ++ IntPtr.Pinned 0.
            if x.collectLogs then
                let infoBufferSize = 1024
                let infoBuffer = System.Text.StringBuilder(infoBufferSize) //TODO                                
                yield CUjitOption.CU_JIT_INFO_LOG_BUFFER ++ IntPtr.Pinned infoBuffer
                yield CUjitOption.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES ++ IntPtr.Pinned infoBufferSize
                let errorBufferSize = 1024
                let errorBuffer = System.Text.StringBuilder(errorBufferSize)
                yield CUjitOption.CU_JIT_ERROR_LOG_BUFFER ++ IntPtr.Pinned errorBuffer
                yield CUjitOption.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES ++ IntPtr.Pinned errorBufferSize
        |] |> Array.unzip3

#nowarn "9"
module Memory = 

    [<Struct>]
    type float16(v: float) = 
        member x.Float = v

    type CudaArrayChannels = 
        | _1 = 1 | _2 = 2 | _4 = 4

    type CudaArray<'elementType> private (cuArray:CUarray, config: CUDA_ARRAY3D_DESCRIPTOR) =
        static let arrayTypes = 
            Map [
                typeof<byte>.GUID, CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8
                typeof<sbyte>.GUID, CUarray_format.CU_AD_FORMAT_SIGNED_INT8
                typeof<uint16>.GUID, CUarray_format.CU_AD_FORMAT_UNSIGNED_INT16
                typeof<int16>.GUID, CUarray_format.CU_AD_FORMAT_SIGNED_INT16
                typeof<uint32>.GUID, CUarray_format.CU_AD_FORMAT_UNSIGNED_INT32
                typeof<int>.GUID, CUarray_format.CU_AD_FORMAT_SIGNED_INT32
                typeof<float16>.GUID, CUarray_format.CU_AD_FORMAT_HALF
                typeof<float>.GUID, CUarray_format.CU_AD_FORMAT_FLOAT
            ]
        static member private CreateArrayInternal(width, height, depth, ?ch, ?flags) = 
            let ch = defaultArg ch CudaArrayChannels._1
            let flags = defaultArg flags CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_NONE
            let mutable cuArray = CUarray.Zero
            let mutable config = 
                CUDA_ARRAY3D_DESCRIPTOR(
                    Width = width, Height = height, Depth = depth, Format = arrayTypes.[typeof<'elementType>.GUID],
                    NumChannels = int ch, Flags = flags
                )
            let ptr, dispose = IntPtr.Pinned config
            let result = cuArray3DCreate(&cuArray, NativeInterop.NativePtr.ofNativeInt(ptr))  
            dispose()            
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuArray3DCreate", result)
            new CudaArray<'elementType>(cuArray, config)
        static member private GenerateFlags(?textureGatherSupprt, ?surfaceSupport, ?flags) = 
            match flags, textureGatherSupprt, surfaceSupport with
            | Some flags, Some true, Some true ->
                flags ||| CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_SURFACE_LDST
                ||| CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_TEXTURE_GATHER
            | None, Some true, Some true -> 
                CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_TEXTURE_GATHER
                ||| CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_SURFACE_LDST
            | Some flags, _, Some true -> flags ||| CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_SURFACE_LDST
            | None, _, Some true -> CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_SURFACE_LDST
            | Some flags, Some true, _ -> flags ||| CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_TEXTURE_GATHER
            | None, Some true, _ -> CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_TEXTURE_GATHER
            | Some flags, _, _ -> flags
            | None, _, _ -> CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_NONE                                        
        static member Create1D(width, ?ch, ?textureGatherSupprt, ?surfaceSupport) = 
            CudaArray<'elementType>.CreateArrayInternal(width, 0u, 0u, ?ch = ch, 
                flags = CudaArray<'elementType>.GenerateFlags(?textureGatherSupprt=textureGatherSupprt, ?surfaceSupport=surfaceSupport))
        static member Create2D(width, height, ?ch, ?textureGatherSupprt, ?surfaceSupport) = 
            CudaArray<'elementType>.CreateArrayInternal(width, height, 0u, ?ch = ch, 
                flags = CudaArray<'elementType>.GenerateFlags(?textureGatherSupprt=textureGatherSupprt, ?surfaceSupport=surfaceSupport))
        static member Create3D(width, height, depth, ?ch, ?textureGatherSupprt, ?surfaceSupport) = 
            CudaArray<'elementType>.CreateArrayInternal(width, height, depth, ?ch = ch, 
                flags = CudaArray<'elementType>.GenerateFlags(?textureGatherSupprt=textureGatherSupprt, ?surfaceSupport=surfaceSupport))
        static member CreateLayered1D(width, numLayers, ?ch, ?textureGatherSupprt, ?surfaceSupport) = 
            CudaArray<'elementType>.CreateArrayInternal(width, 0u, numLayers, ?ch = ch, 
                flags = CudaArray<'elementType>.GenerateFlags(flags = CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_LAYERED, 
                            ?textureGatherSupprt=textureGatherSupprt, ?surfaceSupport=surfaceSupport))
        static member CreateLayered2D(width, height, numLayers, ?ch, ?textureGatherSupprt, ?surfaceSupport) = 
            CudaArray<'elementType>.CreateArrayInternal(width, height, numLayers, ?ch = ch, 
                flags = CudaArray<'elementType>.GenerateFlags(flags = CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_LAYERED, 
                            ?textureGatherSupprt=textureGatherSupprt, ?surfaceSupport=surfaceSupport))
        static member CreateCubemap(size, ?ch, ?textureGatherSupprt, ?surfaceSupport) = 
            CudaArray<'elementType>.CreateArrayInternal(size, size, 6u, ?ch = ch, 
                flags = CudaArray<'elementType>.GenerateFlags(flags = CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_CUBEMAP, 
                            ?textureGatherSupprt=textureGatherSupprt, ?surfaceSupport=surfaceSupport))
        static member CreateLayeredCubemap(size, numCubemaps, ?ch, ?textureGatherSupprt, ?surfaceSupport) = 
            CudaArray<'elementType>.CreateArrayInternal(size, size, 6u*numCubemaps, ?ch = ch, 
                flags = CudaArray<'elementType>.GenerateFlags(flags = (CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_CUBEMAP ||| CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_LAYERED), 
                            ?textureGatherSupprt=textureGatherSupprt, ?surfaceSupport=surfaceSupport))
        interface IDisposable with
            override x.Dispose() = 
                cuArrayDestroy(cuArray) |> ignore
        member x.IsTextureGatherSupported = 
            config.Flags.HasFlag(CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_TEXTURE_GATHER)
        member x.IsSurfaceSupported = 
            config.Flags.HasFlag(CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_SURFACE_LDST)
        member x.IsLayered = 
            config.Flags.HasFlag(CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_LAYERED)
        member x.IsCubemap = 
            config.Flags.HasFlag(CUDA_ARRAY3D_DESCRIPTOR_FLAGS.CUDA_ARRAY3D_CUBEMAP)
        member x.NumberOfChannels = config.NumChannels
        member x.Width = config.Width
        member x.Height = config.Height
        member x.Depth = config.Depth
        //member x.CopyTo(arr:CudaArray<'elementType>, ?thisOffset, ?) = 
            

    type internal DevicePtrSource = Default | Ipc | MappedPinned of IntPtr*CU_MEMHOSTALLOC_FLAGS | UnifiedMemory | IntPtrCasted | Pitch of uint32*uint32*uint32
    
    type CoalescedMemorySize = 
        | _4 = 4 | _8 = 8 | _16 = 16
        
    type IntPtr with        
        member p.LockPages(size, ?mapped, ?portable) = 
            let flags = enum<CU_MEMHOSTREGISTER_FLAGS> 0
            let flags = 
                match mapped with
                | Some true -> flags ||| CU_MEMHOSTREGISTER_FLAGS.CU_MEMHOSTREGISTER_DEVICEMAP
                | _ -> flags
            let flags = 
                match portable with
                | Some true -> flags ||| CU_MEMHOSTREGISTER_FLAGS.CU_MEMHOSTREGISTER_PORTABLE
                | _ -> flags
            let result = cuMemHostRegister(p, size, flags)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemHostRegister", result)   
            fun () -> (cuMemHostUnregister(p) |> ignore)
        static member AllocLockedPages(size, ?mapped, ?writeCombined, ?portableForAllContexts) = 
            let flags = enum<CU_MEMHOSTALLOC_FLAGS> 0
            let flags = 
                match mapped with
                | Some true -> flags ||| CU_MEMHOSTALLOC_FLAGS.CU_MEMHOSTALLOC_DEVICEMAP
                | _ -> flags
            let flags = 
                match writeCombined with
                | Some true -> flags ||| CU_MEMHOSTALLOC_FLAGS.CU_MEMHOSTALLOC_WRITECOMBINED
                | _ -> flags
            let flags = 
                match portableForAllContexts with
                | Some true -> flags ||| CU_MEMHOSTALLOC_FLAGS.CU_MEMHOSTALLOC_PORTABLE
                | _ -> flags                
            let mutable hostPtr = IntPtr.Zero
            let result = cuMemHostAlloc(&hostPtr, size, flags)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemHostAlloc", result)   
            let ptr = hostPtr
            hostPtr, flags, fun () -> cuMemFreeHost(ptr)

    type DevicePtr private (devPtr: CUdeviceptr, initialSize:uint32, source: DevicePtrSource) as x =
        static let mutable pointers: Map<CUdeviceptr, DevicePtr> = Map.empty
        do if source <> IntPtrCasted then pointers <- Map.add devPtr x pointers
        static member FromPointer(p:IntPtr) = 
            new DevicePtr(p.ToInt64() |> uint32, 0u, IntPtrCasted)
        static member FromIpc(remoteHandle: int64) = 
            let handle = CUipcMemHandle.op_Explicit(remoteHandle)
            let mutable ptr = CUdeviceptr.MinValue
            let result = cuIpcOpenMemHandle(&ptr, handle, CU_IPC_MEM_FLAGS.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuIpcOpenMemHandle", result)
            new DevicePtr(ptr, 0u, DevicePtrSource.Ipc)
        static member AllocMappedPinned(size, ?writeCombined, ?portableForAllContexts) =
            let hostPtr, flags, _ = IntPtr.AllocLockedPages(size, defaultArg writeCombined false, defaultArg portableForAllContexts false) 
            let mutable devPtr = CUdeviceptr.MinValue
            let result = cuMemHostGetDevicePointer(&devPtr, hostPtr, 0u)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemHostGetDevicePointer", result)
            new DevicePtr(devPtr, size, MappedPinned (hostPtr, flags))
        static member Alloc(size) = 
            let mutable devPtr = CUdeviceptr.MinValue
            let result = cuMemAlloc(&devPtr, size)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemAlloc", result)
            new DevicePtr(devPtr, size, DevicePtrSource.Default)
        static member AllocManagedHostAttached(size) =
            let mutable devPtr = CUdeviceptr.MinValue
            let result = cuMemAllocManaged(&devPtr, size, CUmemAttach_flags.CU_MEM_ATTACH_HOST)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemAllocManaged", result)
            new DevicePtr(devPtr, size, DevicePtrSource.UnifiedMemory)
        static member AllocManagedGlobalAttached(size) =
            let mutable devPtr = CUdeviceptr.MinValue
            let result = cuMemAllocManaged(&devPtr, size, CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemAllocManaged", result)
            new DevicePtr(devPtr, size, DevicePtrSource.UnifiedMemory)
        static member AllocPitch(widthInBytes, rowNumber, size: CoalescedMemorySize) = 
            let mutable devPtr = CUdeviceptr.MinValue
            let mutable size = 0u
            let result = cuMemAllocPitch(&devPtr, &size, widthInBytes, rowNumber, uint32 size)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemAllocPitch", result)
            new DevicePtr(devPtr, size*rowNumber, Pitch (widthInBytes, rowNumber, size))
        static member DisposeAll = 
            pointers |> Map.iter(fun _ p -> p.Free())
            pointers <- Map.empty
        interface IDisposable with
            override x.Dispose() = 
                match source with
                | IntPtrCasted -> CUresult.CUDA_SUCCESS
                | DevicePtrSource.Ipc -> cuIpcCloseMemHandle(devPtr)
                | MappedPinned (hostPtr, _) -> cuMemFreeHost(hostPtr)
                | _ -> cuMemFree(devPtr) 
                |> ignore
                pointers <- Map.remove devPtr pointers
        member x.Free() = (x :> IDisposable).Dispose()
        member internal x.CuDevicePtr = devPtr
        member x.Pitch = 
            match source with
            | Pitch (_, _, size) -> size
            | _ -> 0u
        member x.AllocatedSize = initialSize
        member x.GetIpc() = 
            let mutable handle = CUipcMemHandle.Zero
            let result = cuIpcGetMemHandle(&handle, devPtr)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuIpcGetMemHandle", result)
            int64 handle
        member x.GetBaseDevicePtr() = 
            let mutable basePtr = CUdeviceptr.MinValue
            let mutable sz = 0u
            let result = cuMemGetAddressRange(&basePtr, &sz, devPtr)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemGetAddressRange", result)
            match Map.tryFind basePtr pointers with
            | Some ptr -> ptr
            | None -> 
                let ptr = new DevicePtr(basePtr, sz, source)
                pointers <- Map.add basePtr ptr pointers
                ptr
        member x.IsIpc = source = Ipc
        member x.IsMappedPinned = match source with | MappedPinned _ -> true | _ -> false
        member x.IsPinnedWriteCombined =
            match source with 
            | MappedPinned (_, flags) -> flags.HasFlag(CU_MEMHOSTALLOC_FLAGS.CU_MEMHOSTALLOC_WRITECOMBINED)
            | _ -> false
        member x.IsPinnedPortable =
            match source with 
            | MappedPinned (_, flags) -> flags.HasFlag(CU_MEMHOSTALLOC_FLAGS.CU_MEMHOSTALLOC_PORTABLE)
            | _ -> false
        member x.IsInUnifiedMemory = source = UnifiedMemory
        member x.IsPitched = match source with | Pitch _ -> true | _ -> false
        member x.Pointer = 
            match source with
            | IntPtrCasted -> IntPtr.op_Explicit(int64 devPtr)
            | _ -> IntPtr.Zero
        member x.CopyTo(ptr:DevicePtr, size) = 
            let result = cuMemcpy(ptr.CuDevicePtr, x.CuDevicePtr, size)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemcpy", result)
        member x.CopyTo(ptr: IntPtr, size) = 
            let result = cuMemcpyDtoH(ptr, x.CuDevicePtr, size)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemcpyDtoH", result)      
        member x.Set(v:uint32, ?elementsCount) =
            match source with
            | Pitch (width, height, pitch) ->
                let result = cuMemsetD2D32(devPtr, pitch, v, width / 4u, height)
                if result <> CUresult.CUDA_SUCCESS then
                    raise <| CudaException("cuMemsetD2D32", result) 
            | _ -> 
                let result = cuMemsetD32(devPtr, v, defaultArg elementsCount initialSize)
                if result <> CUresult.CUDA_SUCCESS then
                    raise <| CudaException("cuMemsetD32", result) 
        member x.Set(v:uint16, ?elementsCount) =
            match source with
            | Pitch (width, height, pitch) ->
                let result = cuMemsetD2D16(devPtr, pitch, v, width / 2u, height)
                if result <> CUresult.CUDA_SUCCESS then
                    raise <| CudaException("cuMemsetD2D16", result) 
            | _ -> 
                let result = cuMemsetD16(devPtr, v, defaultArg elementsCount initialSize)
                if result <> CUresult.CUDA_SUCCESS then
                    raise <| CudaException("cuMemsetD16", result)  
        member x.Set(v:byte, ?elementsCount) =
            match source with
            | Pitch (width, height, pitch) ->
                let result = cuMemsetD2D8(devPtr, pitch, v, width, height)
                if result <> CUresult.CUDA_SUCCESS then
                    raise <| CudaException("cuMemsetD2D8", result) 
            | _ -> 
                let result = cuMemsetD8(devPtr, v, defaultArg elementsCount initialSize)
                if result <> CUresult.CUDA_SUCCESS then
                    raise <| CudaException("cuMemsetD8", result)  
        member internal x.Source = source

    type IntPtr with
        member x.CopyTo(ptr: DevicePtr, size) = 
            let result = cuMemcpyHtoD(ptr.CuDevicePtr, x, size)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemcpyHtoD", result)  
type JitLinker internal (linkerState, disposes: list<_>) = 
    let mutable nativeDisposes = disposes
    member private x.ClearJit() = 
            cuLinkDestroy(linkerState) |> ignore
            nativeDisposes |> List.iter(fun dispose -> dispose())
    member x.AddCubinFile(path) = 
        let result = cuLinkAddFile(linkerState, CUjitInputType.CU_JIT_INPUT_CUBIN, path, 0u, null, null)
        if result <> CUresult.CUDA_SUCCESS then 
            raise <| CudaException("cuLinkAddFile", result)
        x
    member x.AddCubin(data: byte[], ?name:string) = 
        let dataLength = data.Length
        let data, dispose = IntPtr.Pinned(data)
        nativeDisposes <- dispose::nativeDisposes
        let result = cuLinkAddData(linkerState, CUjitInputType.CU_JIT_INPUT_CUBIN, data, dataLength, defaultArg name null, 0u, null, null)
        if result <> CUresult.CUDA_SUCCESS then 
            raise <| CudaException("cuLinkAddData", result)
        x        
    member private x.AddFile(fileType, path, ?options: JitOptions) = 
        let size, keys, values = 
            match options with
            | None -> 0u, null, null
            | Some options -> 
                let keys, values, disposes = options.ToArrays()
                nativeDisposes <- disposes |> Array.fold(fun acc dispose -> dispose::acc) nativeDisposes
                uint32 keys.Length, keys, values
        let result = cuLinkAddFile(linkerState, fileType, path, 0u, null, null)
        if result <> CUresult.CUDA_SUCCESS then 
            raise <| CudaException("cuLinkAddFile", result)
        x
    member private x.Add(fileType, data:string, ?options: JitOptions, ?name) = 
        let dataLength = data.Length
        let data, dispose = IntPtr.Pinned data
        let size, keys, values = 
            match options with
            | None -> 0u, null, null
            | Some options -> 
                let keys, values, disposes = options.ToArrays()
                nativeDisposes <- disposes |> Array.fold(fun acc dispose -> dispose::acc) (dispose::nativeDisposes)
                uint32 keys.Length, keys, values
        let result = cuLinkAddData(linkerState, fileType, data, dataLength, defaultArg name null, size, keys, values)
        if result <> CUresult.CUDA_SUCCESS then 
            raise <| CudaException("cuLinkAddData", result)
        x
    member x.AddPtxFile(path, ?options: JitOptions) = 
        x.AddFile(CUjitInputType.CU_JIT_INPUT_PTX, path, ?options = (options |> Option.map(fun opt -> {opt with fallback=None})))
    member x.AddPtx(text:string, ?options, ?name) = 
        x.Add(CUjitInputType.CU_JIT_INPUT_PTX, text, ?options=options, ?name=name)
    member x.AddFatbinanyFile(path, ?options: JitOptions) = 
        x.AddFile(CUjitInputType.CU_JIT_INPUT_FATBINARY, path, ?options = options)
    member x.AddObjectFile(path, ?options: JitOptions) = 
        x.AddFile(CUjitInputType.CU_JIT_INPUT_OBJECT, path, ?options = options)
    member x.AddLibraryFile(path, ?options: JitOptions) = 
        x.AddFile(CUjitInputType.CU_JIT_INPUT_LIBRARY, path, ?options = options)    
    member x.Complete(?doNotDispose) = 
        let doNotDispose = defaultArg doNotDispose false
        let mutable resultPtr = IntPtr.Zero
        let mutable size = 0
        let result = cuLinkComplete(linkerState, &resultPtr, &size)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuLinkComplete", result)        
        let mutable cuModule = CUmodule.Zero
        let result = cuModuleLoadData(&cuModule, resultPtr)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuModuleLoadData", result)        
        if not doNotDispose then x.ClearJit()
        new Module(cuModule)
    member x.ToBytes(?doNotDispose) = 
        let doNotDispose = defaultArg doNotDispose false
        let mutable resultPtr = IntPtr.Zero
        let mutable size = 0
        let result = cuLinkComplete(linkerState, &resultPtr, &size)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuLinkComplete", result)   
        let res = Array.zeroCreate<byte> size
        Marshal.Copy(resultPtr, res, 0, size)
        res
and Module internal (cuModule:CUmodule) =
    //let mutable cuModule = CUmodule.Zero    
    static member Jit(options: JitOptions) =        
        let mutable linkerState = CUlinkState.Zero
        let keys, values, disposes = options.ToArrays()
        let nativeDisposes = disposes |> Array.fold(fun acc dispose -> dispose::acc) []
        let result = cuLinkCreate(uint32 keys.Length, keys, values, &linkerState)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuLinkCreate", result)
        JitLinker(linkerState, nativeDisposes)
    static member Load(file:string) = 
        let mutable cuModule = CUmodule.Zero
        let result = cuModuleLoad(&cuModule, file)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuModuleLoad", result)
        new Module(cuModule)
    static member LoadFromData(text:string) = 
        let mutable cuModule = CUmodule.Zero
        let ptr, dispose = IntPtr.Pinned text
        let result = cuModuleLoadData(&cuModule, ptr)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuModuleLoadData", result)
        dispose()
        new Module(cuModule)    
    static member LoadFromData(bytes:byte[]) = 
        let mutable cuModule = CUmodule.Zero
        let ptr, dispose = IntPtr.Pinned bytes
        let result = cuModuleLoadData(&cuModule, ptr)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuModuleLoadData", result)
        dispose()
        new Module(cuModule)              
    static member LoadFromData(text:string, options:JitOptions) = 
        let mutable cuModule = CUmodule.Zero
        let ptr, dispose = IntPtr.Pinned text
        let keys, values, disposes = options.ToArrays()
        let result = cuModuleLoadDataEx(&cuModule, ptr, uint32 keys.Length, keys, values)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuModuleLoadDataEx", result)
        disposes |> Array.iter(fun dispose -> dispose())
        dispose()
        new Module(cuModule)
    static member LoadFromData(bytes: byte[], ?options:JitOptions) = 
        let options = defaultArg options JitOptions.Default
        let mutable cuModule = CUmodule.Zero
        let ptr, dispose = IntPtr.Pinned bytes
        let keys, values, disposes = options.ToArrays()
        let result = cuModuleLoadDataEx(&cuModule, ptr, uint32 keys.Length, keys, values)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuModuleLoadDataEx", result)
        disposes |> Array.iter(fun dispose -> dispose())
        dispose()
        new Module(cuModule)
    static member LoadFromFatBinary(fatBin: byte[]) =
        let mutable cuModule = CUmodule.Zero
        let ptr, dispose = IntPtr.Pinned fatBin
        let result = cuModuleLoadFatBinary(&cuModule, ptr)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuModuleLoadDataEx", result)
        dispose()
        new Module(cuModule)
    interface IDisposable with
        override x.Dispose() = 
            cuModuleUnload(cuModule) |> ignore
    member x.Unload() = (x :> IDisposable).Dispose()
    member x.GetFunction(name) = 
        let mutable cuFunc = CUfunction.Zero
        let result = cuModuleGetFunction(&cuFunc, cuModule, name)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuModuleGetFunction", result)  
        cuFunc      
    member x.GetGlobal(name) = 
        let mutable cuPtr = CUdeviceptr.MinValue
        let mutable size = 0
        let result = cuModuleGetGlobal(&cuPtr, &size, cuModule, name)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuModuleGetGlobal", result)  
        cuPtr, size 
    member x.GetSurface(name) = 
        let mutable surfRef = CUsurfref.Zero
        let result = cuModuleGetSurfRef(&surfRef, cuModule, name)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuModuleGetSurfRef", result)  
        surfRef     
    member x.GetTexture(name) = 
        let mutable texRef = CUtexref.Zero
        let result = cuModuleGetTexRef(&texRef, cuModule, name)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuModuleGetTexRef", result)  
        texRef         
and Kernel private (func:CUfunction) as x =
    let mutable attributes: Map<CUfunction_attribute, int> = Map.empty
    let mutable gridDim = (0u, 0u, 0u)
    let mutable blockDim = (0u, 0u, 0u)
    let mutable sharedMemorySize = 0u
    let mutable stream = CudaStream.Null
    let mutable disposes: (unit -> unit)[] = Array.empty
    let mutable bindedParameters: obj list = []
    let getAttribute attribute = 
        match Map.tryFind attribute attributes with
        | None ->
            let mutable v = 0
            let result = cuFuncGetAttribute(&v, attribute, func)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuFuncGetAttribute", result)
            attributes <- Map.add attribute v attributes
            v
        | Some attribute -> attribute
    let preferCacheConfig config = 
        let result = cuFuncSetCacheConfig(func, config)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuFuncSetCacheConfig", result)
        x
    member x.RequiredThreadsPerBlock = getAttribute CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    member x.StaticalyAllocatedSharedMemory = getAttribute CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
    member x.ConstantMemory = getAttribute CUfunction_attribute.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
    member x.LocalMemory = getAttribute CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
    member x.RegistersUsage = getAttribute CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS
    member x.PtxVersion = getAttribute CUfunction_attribute.CU_FUNC_ATTRIBUTE_PTX_VERSION
    member x.BinaryVersion = getAttribute CUfunction_attribute.CU_FUNC_ATTRIBUTE_BINARY_VERSION
    member x.IsCacheModeCA = 
        let mutable v = 0
        let result = cuFuncGetAttribute(&v, CUfunction_attribute.CU_FUNC_CACHE_MODE_CA, func)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuFuncGetAttribute", result)
        v = 1
    member x.PreferDefaultCacheConfig() = preferCacheConfig CUfunc_cache.CU_FUNC_CACHE_PREFER_NONE
    member x.PreferSharedMemory() = preferCacheConfig CUfunc_cache.CU_FUNC_CACHE_PREFER_SHARED
    member x.PreferL1() = preferCacheConfig CUfunc_cache.CU_FUNC_CACHE_PREFER_L1
    member x.PreferSharedMemoryEqualL1() = preferCacheConfig CUfunc_cache.CU_FUNC_CACHE_PREFER_EQUAL
    member x.SharedMemoryBankSize(size:int) =
        let config = 
            match size with
            | 4 -> CUsharedconfig.CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE 
            | 8 -> CUsharedconfig.CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE
            | _ -> CUsharedconfig.CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE
        let result = cuFuncSetSharedMemConfig(func, config)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuFuncSetSharedMemConfig", result)
        x
    member this.Grid(x, ?y, ?z) = 
        gridDim <- (x, defaultArg y 0u, defaultArg z 0u) 
        this
    member this.Block(x, ?y, ?z) = 
        blockDim <- (x, defaultArg y 0u, defaultArg z 0u) 
        this
    member this.SharedMemory(size) = 
        sharedMemorySize <- size
        this
    member x.Stream(s: CudaStream) = 
        stream <- s
        x
    member x.Params(parameters:seq<obj>) = 
        bindedParameters <- bindedParameters @ (parameters |> Seq.toList)
        x
    member x.Params([<ParamArray>]parameters:obj list) = 
        bindedParameters <- bindedParameters @ parameters
        x
    member x.Params<'T>(parameters:seq<'T>) = 
        bindedParameters <- bindedParameters @ (parameters |> Seq.map(fun p -> p :> obj) |> Seq.toList)
        x
    member x.Params<'T>([<ParamArray>]parameters:'T list) = 
        bindedParameters <- bindedParameters @ (parameters |> List.map(fun p -> p :> obj))
        x
    member x.Launch(parameters: seq<Object>) = 
        let gridX, gridY, gridZ = gridDim
        let blockX, blockY, blockZ = blockDim
        let ptrs, disp = 
            Seq.concat([bindedParameters |> List.toSeq; parameters]) 
            |> Seq.map(fun p -> IntPtr.Pinned p) |> Seq.toArray |> Array.unzip
        let result = cuLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemorySize, stream.CuStream, ptrs, null) 
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuLaunchKernel", result)
        disposes <- disp //!!!! TODO - dispose
    member x.Launch([<ParamArray>]parameters:Object[]) = x.Launch(parameters)
and CudaStream private (stream: CUstream) as x = 
    static let mutable existedStreams: Map<CUstream, CudaStream> = Map.empty
    static let defaultStream = new CudaStream(CUstream.Zero)
    let mutable flags: Option<CU_STREAM_FLAGS> = None
    let getFlags() = 
        match flags with
        | None ->
            let mutable v = 0
            let result = cuStreamGetFlags(stream, &v)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuStreamGetFlags", result)
            flags <- Some(enum<CU_STREAM_FLAGS> v)
            flags.Value
        | Some v -> v
    do existedStreams <- Map.add stream x existedStreams    
    member internal x.CuStream = stream
    static member Null = defaultStream
    static member private CreateInternal(flags: CU_STREAM_FLAGS, ?priority) = 
        match priority with
        | None ->
            let mutable stream = CUstream.Zero
            let result = cuStreamCreate(&stream, flags)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuStreamCreate", result)
            new CudaStream(stream)
        | Some priority ->
            let mutable stream = CUstream.Zero
            let result = cuStreamCreateWithPriority(&stream, flags, priority)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuStreamCreateWithPriority", result)
            new CudaStream(stream)   
    static member private Create(stream) =
        match Map.tryFind stream existedStreams with
        | None -> raise <| ArgumentException("Specified stream was not found")
        | Some stream -> stream
    static member Create(?priority) = CudaStream.CreateInternal(CU_STREAM_FLAGS.CU_STREAM_DEFAULT, ?priority = priority)
    static member CreateNonBlocking(?priority) = 
        CudaStream.CreateInternal(CU_STREAM_FLAGS.CU_STREAM_NON_BLOCKING, ?priority = priority)
    interface IDisposable with
        override x.Dispose() = 
            if stream <> CUstream.Zero then
                cuStreamDestroy(stream) |> ignore
                existedStreams <- Map.remove stream existedStreams
    member x.Priority = 
        let mutable v = 0
        let result = cuStreamGetPriority(stream, &v)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuStreamGetPriority", result)
        v
    member x.Dispose() = (x :> IDisposable).Dispose()
    member x.Callback(func:CudaStream -> unit, ?errHandler: CudaStream*CudaException -> unit) = 
        let handler = 
            match errHandler with
            | None -> 
                fun (stream: CUstream, cuResult, data: IntPtr) -> 
                    if cuResult = CUresult.CUDA_SUCCESS then 
                        func(CudaStream.Create(stream))
                    else raise <| CudaException("stream callback", cuResult)
            | Some handler ->
                fun (stream, cuResult, data) -> 
                    let stream = CudaStream.Create(stream)
                    if cuResult = CUresult.CUDA_SUCCESS then func(stream)
                    else handler(stream, CudaException("stream callback", cuResult))
        let result = cuStreamAddCallback(stream, CUstreamCallback(handler), IntPtr.Zero, 0u)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuStreamAddCallback", result)
        x
    member x.Callback<'data>(func:CudaStream*'data -> unit, data:'data, ?errHandler: CudaStream*CudaException*'data -> unit) = 
        let ptr, dispose = IntPtr.Pinned data
        let handler = 
            match errHandler with
            | None -> 
                fun (stream: CUstream, cuResult, data) -> 
                    if cuResult = CUresult.CUDA_SUCCESS then 
                        let data = GCHandle.FromIntPtr(data).Target :?> 'data
                        dispose()
                        func(CudaStream.Create(stream), data)
                    else raise <| CudaException("stream callback", cuResult)
            | Some handler ->
                fun (stream, cuResult, data) -> 
                    let stream = CudaStream.Create(stream)
                    let data = GCHandle.FromIntPtr(data).Target :?> 'data
                    dispose()
                    if cuResult = CUresult.CUDA_SUCCESS then func(stream, data)
                    else handler(stream, CudaException("stream callback", cuResult), data)
        let result = cuStreamAddCallback(stream, CUstreamCallback(handler), ptr, 0u)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuStreamAddCallback", result)
        x  
    member x.Launch(func: Kernel, [<ParamArray>]parameters:Object[]) = 
        func.Stream(x).Launch(parameters)
        x
    member x.Copy(source: Memory.DevicePtr, dest: Memory.DevicePtr, size) = 
        let result = cuMemcpyAsync(dest.CuDevicePtr, source.CuDevicePtr, size, stream)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuMemcpyAsync", result)
        x 
    member x.Copy(source: Memory.DevicePtr, dest: IntPtr, size) = 
        let result = cuMemcpyDtoHAsync(dest, source.CuDevicePtr, size, stream)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuMemcpyDtoHAsync", result)
        x         
    member x.Copy(source: IntPtr, dest: Memory.DevicePtr, size) = 
        let result = cuMemcpyHtoDAsync(dest.CuDevicePtr, source, size, stream)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuMemcpyHtoDAsync", result)
        x    
    member x.SetMemory(devPtr: Memory.DevicePtr, v:uint32, ?elementsCount) =
        match devPtr.Source with
        | Memory.DevicePtrSource.Pitch (width, height, pitch) ->
            let result = cuMemsetD2D32Async(devPtr.CuDevicePtr, pitch, v, width / 4u, height, stream)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemsetD2D32Async", result) 
        | _ -> 
            let result = cuMemsetD32Async(devPtr.CuDevicePtr, v, defaultArg elementsCount devPtr.AllocatedSize, stream)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemsetD32Async", result) 
    member x.SetMemory(devPtr: Memory.DevicePtr, v:uint16, ?elementsCount) =
        match devPtr.Source with
        | Memory.DevicePtrSource.Pitch (width, height, pitch) ->
            let result = cuMemsetD2D16Async(devPtr.CuDevicePtr, pitch, v, width / 2u, height, stream)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemsetD2D16Async", result) 
        | _ -> 
            let result = cuMemsetD16Async(devPtr.CuDevicePtr, v, defaultArg elementsCount devPtr.AllocatedSize, stream)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemsetD16Async", result) 
    member x.SetMemory(devPtr: Memory.DevicePtr, v:byte, ?elementsCount) =
        match devPtr.Source with
        | Memory.DevicePtrSource.Pitch (width, height, pitch) ->
            let result = cuMemsetD2D8Async(devPtr.CuDevicePtr, pitch, v, width, height, stream)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemsetD2D8Async", result) 
        | _ -> 
            let result = cuMemsetD8Async(devPtr.CuDevicePtr, v, defaultArg elementsCount devPtr.AllocatedSize, stream)
            if result <> CUresult.CUDA_SUCCESS then
                raise <| CudaException("cuMemsetD8Async", result)            
    member x.AttachManagedDevicePtr(ptr: Memory.DevicePtr) = 
        let result = cuStreamAttachMemAsync(stream, ptr.CuDevicePtr, 0, CUmemAttach_flags.CU_MEM_ATTACH_SINGLE)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuStreamAttachMemAsync", result)
        x
    member x.IsNonBlocking = 
        let flags = getFlags()
        flags.HasFlag(CU_STREAM_FLAGS.CU_STREAM_NON_BLOCKING)
    member x.IsReady = 
        let result = cuStreamQuery(stream)
        match result with
        | CUresult.CUDA_ERROR_NOT_READY -> false
        | CUresult.CUDA_SUCCESS -> true
        | result -> raise <| CudaException("cuStreamQuery", result)
    member x.Synchronize() = 
        let result = cuStreamSynchronize(stream)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuStreamSynchronize", result)
        x
    member x.RecordEvent(ev: CudaEvent) = 
        ev.RecordTo(x) |> ignore
        x
    member x.Wait(ev:CudaEvent) = 
        let result = cuStreamWaitEvent(stream, ev.CuEvent, 0u)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuStreamWaitEvent", result)
        x
and CudaEvent private (ev: CUevent) = 
    static let createWithFlags(flags) = 
        let mutable ev = CUevent.Zero
        let result = cuEventCreate(&ev, flags)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuEventCreate", result)
        new CudaEvent(ev)
    static member FromIpc(remoteHandle: int64) = 
        let handle = CUipcEventHandle.op_Explicit(remoteHandle)
        let mutable ev = CUevent.Zero
        let result = cuIpcOpenEventHandle(&ev, handle)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuIpcOpenEventHandle", result)
        new CudaEvent(ev)
    static member Create(?enableTiming, ?isBlocking, ?forIpc) = 
        let flags =
            match forIpc, enableTiming, isBlocking with
            | Some true, _, Some true -> CU_EVENT_FLAGS.CU_EVENT_DISABLE_TIMING ||| CU_EVENT_FLAGS.CU_EVENT_INTERPROCESS ||| CU_EVENT_FLAGS.CU_EVENT_BLOCKING_SYNC
            | Some true, _, _ -> CU_EVENT_FLAGS.CU_EVENT_DISABLE_TIMING ||| CU_EVENT_FLAGS.CU_EVENT_INTERPROCESS
            | _, Some true, Some true -> CU_EVENT_FLAGS.CU_EVENT_BLOCKING_SYNC
            | _, _, Some true -> CU_EVENT_FLAGS.CU_EVENT_DISABLE_TIMING ||| CU_EVENT_FLAGS.CU_EVENT_BLOCKING_SYNC
            | _, Some true, _ -> CU_EVENT_FLAGS.CU_EVENT_DEFAULT
            | _, _, _ -> CU_EVENT_FLAGS.CU_EVENT_DISABLE_TIMING
        createWithFlags(flags)
    interface IDisposable with
        override x.Dispose() = 
            cuEventDestroy(ev) |> ignore
    member internal x.CuEvent = ev
    static member ElapsedTime(startEv: CudaEvent, endEv: CudaEvent) =
        let mutable time = 0.
        let result = cuEventElapsedTime(&time, startEv.CuEvent, endEv.CuEvent)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuEventElapsedTime", result)
        time
    member x.RecordTo(stream:CudaStream) = 
        let result = cuEventRecord(x.CuEvent, stream.CuStream)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuEventRecord", result) 
        x 
    member x.IsReady = 
        let result = cuEventQuery(ev)
        match result with
        | CUresult.CUDA_ERROR_NOT_READY -> false
        | CUresult.CUDA_SUCCESS -> true
        | result -> raise <| CudaException("cuEventQuery", result)  
    member x.GetIpc() = 
        let mutable handle = CUipcEventHandle.Zero
        let result = cuIpcGetEventHandle(&handle, ev)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuIpcGetEventHandle", result)
        int64 handle
    member x.Synchronize() =
        let result = cuEventSynchronize(ev)
        if result <> CUresult.CUDA_SUCCESS then
            raise <| CudaException("cuEventSynchronize", result)
        x

module Profiler = 
    open System.Configuration
    let isEnabled = 
        match ConfigurationManager.AppSettings.["cudaProfilerConfig"] with
        | null -> false
        | config -> 
            let configArray = config.Split(';')
            if configArray.Length >= 2 then
                let configFile, outputFile = configArray.[0], configArray.[1]
                let format = 
                    if System.IO.Path.GetExtension(outputFile) = ".csv" then
                        CU_PROFILER_OPTIONS.CU_OUT_CSV
                    else CU_PROFILER_OPTIONS.CU_OUT_KEY_VALUE_PAIR
                let result = cuProfilerInitialize(configFile, outputFile, format)
                match result with
                | CUresult.CUDA_SUCCESS -> true
                | _ -> false
            else false
    let run(func) = 
        if isEnabled then
            cuProfilerStart() |> ignore
            let res = func()
            cuProfilerStop() |> ignore
            res
        else func()
        
