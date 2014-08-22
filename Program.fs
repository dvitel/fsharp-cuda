open System
open System.Runtime.InteropServices
open Cuda.DriverAPI

//let enumerateDevices() =
//    Cuda.Driver.init()
//    for i in 0..Cuda.Device.count.Value-1 do
//        let device = Cuda.Device.getAt i
//        printfn "%d  %s, SM: %d, memory: %d GB" (i+1) <| Cuda.Device.getNameOf(device) 
//            <| Cuda.Device.getAttribute device CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
//            <| (Cuda.Device.getTotalMemoryOf(device) / 1024UL / 1024UL / 1024UL)

[<EntryPoint>]
let main argv =       
    //enumerateDevices()
//    try
//    let mutable a = IntPtr.Zero
//    //let x = Marshal.StringToHGlobalAnsi(a);
//    cuGetErrorString(CUresult.CUDA_ERROR_ASSERT, &a) |> ignore        
//    Console.WriteLine(Marshal.PtrToStringAnsi(a));
//    with e -> printfn "%A" e    
//    Cuda.Device.All |> Seq.iter(fun device -> printfn "%A %A %A" device.Index device.Name device.TotalMemory)
    //cuda.ptx.compileToString <@@ fun (a:_[]) b -> a.[0] + b @@> |> printfn "%s"
(*
compileToString <@@ fun a b -> a + b @@> |> printfn "%s"
compileToString <@@ fun (a:int[]) b (c:int[]) -> c.[0] <- a.[0] + b @@> |> printfn "%s"
*)
    cuda.ptx.compileToString 
        <@@ fun a b -> a + b @@> |> printfn "%s"

    Console.ReadKey() |> ignore    
    

//    printfn "%A" argv
//    if cuInit(0u) = cuResult.CUDA_SUCCESS then
//        let mutable driverVersion = 0
//        match cuDriverGetVersion(&driverVersion) with
//        | cuResult.CUDA_SUCCESS -> printfn "Ok"
//        | error -> printfn "cuda driver version error: %A" error
//    else printfn "init fail"
    0 // return an integer exit code
