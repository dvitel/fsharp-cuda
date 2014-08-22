module cuda.nvcc
open System
open System.Diagnostics

type CompilePhase = | Cuda | Cubin | Ptx | Gpu | Preprocess | GenerateDependecies | Compile | Link | Lib | Run
type Options = {
    compilePhase: CompilePhase
    outputFile: string
    outputDir: string
    preInclude: string seq
    libraries: string seq
    defineMacro: string seq
    undefineMacro: string seq
    includePath: string seq
    libraryPath: string seq

}