module cuda.ptx

//#r @"D:\My projects\CUDA.NET\CUDA.NET\bin\Debug\FSharp.PowerPack.Linq.dll"
open System
open System.Reflection
open Microsoft.FSharp.Reflection
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape
open Microsoft.FSharp.Linq.QuotationEvaluation

//Ptx
type PtxAttribute = ReflectedDefinitionAttribute

//testing Ptx attribute
module kernels = 
    [<Ptx>]
    let test1(a:int[], b:int[], c:int[]) = 
        let d = 5;
        c.[0] <- a.[0] + b.[0]
    [<Ptx>]
    let test2(a:int[,]) =
        a.GetLength(0)
    [<Ptx>]
    let test3() = 0
        //Array2D.in

//get reflected definition from MethodInfo
let getPtxFromMethod acc (func: MethodInfo) = 
    match Expr.TryGetReflectedDefinition(func) with
    | Some expr -> (func, expr)::acc
    | None -> acc
    
//get reflected definitions from Type
let getPtxFromType acc (t: Type) = 
    t.GetMethods(BindingFlags.Public ||| BindingFlags.Static) |> Array.fold(fun acc func ->
        if (func.GetCustomAttributes(typeof<PtxAttribute>) |> Seq.length) > 0 then
            getPtxFromMethod acc func
        else acc) acc  

//get reflected definitions from Assembly
let getPtxFromAssembly acc (assembly: Assembly) =        
    assembly.GetTypes() |> Array.fold(fun acc t ->
        if (t.GetCustomAttributes(typeof<PtxAttribute>) |> Seq.length) > 0 then
            t.GetMethods(BindingFlags.Public ||| BindingFlags.Static) 
            |> Array.fold(getPtxFromMethod) acc
        else getPtxFromType acc t
        ) acc

//get reflected definitions from current Assembly
let getPtxForCurrentAssembly() = 
    getPtxFromAssembly [] (Assembly.GetExecutingAssembly())

// compilation to ptx
type PtxTypeSize = _8 = 8 | _16 = 16 | _32 = 32 | _64 = 64 //sizes in bits
type PtxType = 
    | S of PtxTypeSize //int
    | U of PtxTypeSize //uint
    | F of PtxTypeSize //float
    | B of PtxTypeSize //bit
    | Pred
    | V2 of PtxType
    | V4 of PtxType
with
    override x.ToString() = 
        match x with
        | S v -> sprintf ".s%d" <| int v
        | U v -> sprintf ".u%d" <| int v
        | B v -> sprintf ".b%d" <| int v
        | F PtxTypeSize._8 | F PtxTypeSize._16 -> failwith ".f8 is not supported by ptx. .f16 is not supported by F#"
        | F v -> sprintf ".f%d" <| int v
        | Pred -> ".pred" 
        | V2(V2 _) | V2(V4 _) | V4(V2 _) | V4(V4 _) | V2(Pred) | V4(Pred) -> failwith "ptx foes not support vectors of vectors or vectors of predicates" 
        | V2 t -> sprintf ".v2%s" <| t.ToString()
        | V4 t -> sprintf ".v4%s" <| t.ToString()
    member x.GetAlignment() = 
        match x with
        | S v | U v | F v | B v 
        | V2 (S v) | V2 (U v) | V2 (F v) | V2 (B v)
        | V4 (S v) | V4 (U v) | V4 (F v) | V4 (B v)  -> int v / 8
        | v -> failwithf "Could not detect alignment for specified ptx type: %A" v                                        
    member x.IsVector = 
        match x with
        | V2 _ | V4 _ -> true
        | _ -> false
    member x.IsPredicate = 
        match x with
        | Pred -> true | _ -> false
    member x.ByteSize = 
        match x with
        | S size | U size | F size | B size -> int size / 8
        | V2 t -> 2*t.ByteSize
        | V4 t -> 2*t.ByteSize
        | v -> failwith "Unsupported size %A"

type Type with
    member x.ToPtxType() = 
        let x = if x.IsArray then x.GetElementType() else x
        match x with
        | v when v = typeof<sbyte> -> S PtxTypeSize._8
        | v when v = typeof<System.Int16> -> S PtxTypeSize._16
        | v when v = typeof<int> -> S PtxTypeSize._32
        | v when v = typeof<int64> -> S PtxTypeSize._64
        | v when v = typeof<byte> -> U PtxTypeSize._8
        | v when v = typeof<System.UInt16> -> U PtxTypeSize._16
        | v when v = typeof<uint32> -> U PtxTypeSize._32
        | v when v = typeof<uint64> -> U PtxTypeSize._64
        //.f16 is not supported !!!
        | v when v = typeof<float32> -> F PtxTypeSize._32
        | v when v = typeof<float> -> F PtxTypeSize._64
        | v when FSharpType.IsTuple(v) ->
            let elements = FSharpType.GetTupleElements(v)
            let firstType = elements.[0]
            if Array.TrueForAll(elements, fun el -> el = firstType) then
                let elPtxType = firstType.ToPtxType()
                match elPtxType with
                | V2 _ | V4 _ | Pred -> failwithf "ptx foes not support vectors of vectors or vectors of predicates: %A" v
                | _ when elements.Length = 2 -> V2 elPtxType
                | _ when elements.Length = 4 -> V4 elPtxType
                | _ -> failwithf ".ptx only support vectors of 2 and 4 elements: %A" v
            else failwithf "Ptx does not support vectors of different types: %A" v
        | v -> failwithf "Type %A is not supported for conversion to ptx" v                                        

type MemoryType = | Global | Shared | Const | Local
with 
    override x.ToString() = 
        match x with
        | Global -> ".global"
        | Shared -> ".shared"
        | Const -> ".const"
        | Local -> ".local"

type Param = {
    ptxType: PtxType
    name: string
    pointerTo: MemoryType option
    index: int
    variable: Var
    isProp: bool
    props: Map<string, Param>
}
with
    member x.PtxTypeForRegister = 
        match x.pointerTo with
        | Some _ -> PtxType.U PtxTypeSize._32
        | None -> x.ptxType
    override x.ToString() = 
        match x.pointerTo with
        | None ->
            sprintf ".param %s %s" (x.ptxType.ToString()) x.name
        | Some memoryType ->
            sprintf ".param .u32 .ptr%s.align %d %s" (memoryType.ToString()) (x.ptxType.GetAlignment()) x.name

type PtxSignature() = 
    let mutable parameters = Map.empty
    member x.New(v: Var) = 
        let memoryType, ptxType, props =             
            if v.Type.IsArray then 
                let arrayRank = v.Type.GetArrayRank()
                if arrayRank = 1 then
                    let lengthName = sprintf "%sLength" v.Name
                    let length = //length property
                        {ptxType = PtxType.U PtxTypeSize._32; 
                         pointerTo = None; name = lengthName; 
                         index = parameters.Count; variable = v;
                         isProp = true; props = Map.empty }
                    parameters <- Map.add lengthName length parameters
                    Some Global, v.Type.GetElementType().ToPtxType(), Map [ "Length", length ]
                else 
                    let lengths = //lengths properties
                        Map [for i in 1..arrayRank do
                                let lengthName = sprintf "%sLength%d" v.Name i; 
                                let lengthProp = 
                                    {ptxType = PtxType.U PtxTypeSize._32; 
                                     pointerTo = None; name = lengthName
                                     index = parameters.Count; variable = v;
                                     isProp = true; props = Map.empty 
                                    }
                                parameters <- Map.add lengthName lengthProp parameters
                                yield sprintf "Length%d" i, lengthProp
                                ]
                    Some Global, v.Type.GetElementType().ToPtxType(), lengths
            else None, v.Type.ToPtxType(), Map.empty
        let p = 
            {ptxType = ptxType; pointerTo = memoryType; 
                name = v.Name; index = parameters.Count; variable = v;
                isProp = false; props = props}
        parameters <- Map.add v.Name p parameters
        p
    member x.TryGetParam(v: Var) = Map.tryFind v.Name parameters
    override x.ToString() = 
        let list = 
            parameters |> Map.fold(fun acc _ p -> p::acc) [] |> List.sortBy(fun p -> p.index)
            |> List.map(fun p -> p.ToString())
        String.Join(", ", list)
    member x.AsKernel(name) = 
        sprintf ".entry %s(%s)" name <| x.ToString()
    member x.AsFunc(name) = 
        sprintf ".func %s(%s)" name <| x.ToString()        

type Register = {
    index: int
    ptxType: PtxType       
    variable: string option
    parameter: Param option
    props: Map<string, Register>
}
with 
    member x.Name  =
        let bindedPrefix = 
            match x.variable with
            | Some _ -> "b"
            | None -> ""
        sprintf "%%%sr%s_%d" bindedPrefix (x.ptxType.ToString().Replace(".", "_")) x.index
    override x.ToString() = 
        sprintf ".reg %s %s;" (x.ptxType.ToString()) x.Name

type Registers = {
    free: Register list
    busy: Map<int, Register>
}

type PtxDeclarations(intent) =
    let mutable registers: Map<PtxType, Registers> = Map.empty
    let mutable bindedRegisters: Map<string, Register> = Map.empty
    let mutable bindedRegistersCount: Map<PtxType, (int list)*int> = Map.empty
    let mutable bindingSpaces: Set<PtxType*string> list = []
    let mutable labelsCount = 0
    let spaces = String.replicate intent " "
    member x.NewReg(ptxType) = 
        match Map.tryFind ptxType registers with
        | None -> 
            let reg = {index = 0; ptxType = ptxType; variable = None; parameter = None; props = Map.empty}            
            registers <- Map.add ptxType {free=[]; busy = Map [reg.index, reg]; } registers
            reg
        | Some ({free=[]} as regs) ->
            let reg = {index = regs.busy.Count; ptxType = ptxType; variable = None; parameter = None; props = Map.empty}
            registers <- Map.add ptxType {regs with busy=Map.add reg.index reg regs.busy} registers
            reg
        | Some ({free=reg::freeRegs} as regs) ->
            registers <- Map.add ptxType {regs with free=freeRegs; busy=Map.add reg.index reg regs.busy} registers
            reg
    member x.NewLabel() = 
        let label = sprintf "L_%d" labelsCount
        labelsCount <- labelsCount + 1
        label
    member x.NewBinded(v:Var) = //TODO - add binded to binding spaces
        match Map.tryFind v.Name bindedRegisters with
        | Some reg -> reg
        | None ->
            let ptxType = v.Type.ToPtxType()
            match Map.tryFind ptxType bindedRegistersCount with
            | Some ([], allocCount) -> 
                let reg = {index = allocCount+1; ptxType = ptxType; variable = Some v.Name; parameter = None; props = Map.empty}
                bindedRegisters <- Map.add v.Name reg bindedRegisters
                bindedRegistersCount <- Map.add ptxType ([], allocCount+1) bindedRegistersCount
                reg
            | Some (regIndex::free, allocCount) ->
                let reg = {index = regIndex; ptxType = ptxType; variable = Some v.Name; parameter = None; props = Map.empty}
                bindedRegisters <- Map.add v.Name reg bindedRegisters
                bindedRegistersCount <- Map.add ptxType (free, allocCount) bindedRegistersCount
                reg
            | None ->
                let reg = {index = 0; ptxType = ptxType; variable = Some v.Name; parameter = None; props = Map.empty}
                bindedRegisters <- Map.add v.Name reg bindedRegisters
                bindedRegistersCount <- Map.add ptxType ([], 1) bindedRegistersCount
                reg
    member x.NewBinded(param:Param) = 
        match Map.tryFind param.name bindedRegisters with
        | Some reg -> reg
        | None ->
            let ptxType = param.PtxTypeForRegister
            let props = param.props |> Map.map(fun _ param -> x.NewBinded(param))
            match Map.tryFind ptxType bindedRegistersCount with
            | Some ([], allocCount) -> 
                let reg = {index = allocCount+1; ptxType = ptxType; variable = Some param.name; parameter = Some param; props = props}
                bindedRegisters <- Map.add param.name reg bindedRegisters
                bindedRegistersCount <- Map.add ptxType ([], allocCount+1) bindedRegistersCount
                reg
            | Some (regIndex::free, allocCount) ->
                let reg = {index = regIndex; ptxType = ptxType; variable = Some param.name; parameter = Some param; props = props}
                bindedRegisters <- Map.add param.name reg bindedRegisters
                bindedRegistersCount <- Map.add ptxType (free, allocCount) bindedRegistersCount
                reg
            | None ->
                let reg = {index = 0; ptxType = ptxType; variable = Some param.name; parameter = Some param; props = props}
                bindedRegisters <- Map.add param.name reg bindedRegisters
                bindedRegistersCount <- Map.add ptxType ([], 1) bindedRegistersCount
                reg
    member x.NewBindingSpace() = 
        bindingSpaces <- Set.empty::bindingSpaces
    member x.Unbind(v:Var) = 
        let ptxType = v.Type.ToPtxType()
        let unbind(reg: Register) = 
            bindedRegisters <- Map.remove v.Name bindedRegisters
            match Map.tryFind reg.ptxType bindedRegistersCount with
            | None -> ()
            | Some (free, allocated) -> 
                bindedRegistersCount <- Map.add ptxType (reg.index::free, allocated) bindedRegistersCount
        match bindingSpaces with
        | space::_ when space.Contains(ptxType, v.Name) -> ()
        | _ -> Map.tryFind v.Name bindedRegisters |> Option.iter unbind
    member x.Unbind(reg:Register) =         
        let unbind name = 
            Map.tryFind name bindedRegisters
            |> Option.iter(fun reg ->
                bindedRegisters <- Map.remove name bindedRegisters
                match Map.tryFind reg.ptxType bindedRegistersCount with
                | None -> ()
                | Some (free, allocated) -> 
                    bindedRegistersCount <- Map.add reg.ptxType (reg.index::free, allocated) bindedRegistersCount)
        match bindingSpaces, reg.variable with
        | [], Some name -> unbind name
        | space::_, Some name when space.Contains(reg.ptxType, reg.Name) |> not ->
            unbind name
        | _, _ -> ()     
    member x.UnbindSpace() = 
        match bindingSpaces with
        | [] -> ()
        | space::spaces ->
            space |> Set.iter(fun (ptxType, name) ->
                match Map.tryFind name bindedRegisters with
                | None -> ()
                | Some reg -> 
                    bindedRegisters <- Map.remove name bindedRegisters
                    match Map.tryFind reg.ptxType bindedRegistersCount with
                    | None -> ()
                    | Some (free, allocated) -> 
                        bindedRegistersCount <- Map.add reg.ptxType (reg.index::free, allocated) bindedRegistersCount)
            bindingSpaces <- spaces
    member x.Free([<ParamArray>] regs: Register[]) =      
        regs |> Array.iter(fun reg ->
            match Map.tryFind reg.ptxType registers with
            | None -> ()
            | Some regs ->
                Map.tryFind reg.index regs.busy
                |> Option.iter(fun _ ->
                    registers <- Map.add reg.ptxType {regs with free=reg::regs.free; busy=Map.remove reg.index regs.busy} registers))
    member x.FreeAndAlloc(ptxType, [<ParamArray>] regs: Register[]) = 
        x.Free(regs)
        x.NewReg(ptxType)
    override x.ToString() = 
        let lines = 
            registers 
            |> Map.fold(fun acc ptxType regs -> 
                let count = regs.free.Length + regs.busy.Count 
                let ptxTypeString = ptxType.ToString()                  
                (sprintf "%s.reg %s %%r%s_<%d>;" spaces ptxTypeString (ptxTypeString.Replace(".", "_")) count)::acc
                ) []
        let lines =
            bindedRegistersCount
            |> Map.fold(fun acc ptxType (_, count) -> 
                let ptxTypeString = ptxType.ToString()                  
                (sprintf "%s.reg %s %%br%s_<%d>;" spaces ptxTypeString (ptxTypeString.Replace(".", "_")) count)::acc
                ) lines
        String.Join(Environment.NewLine, lines)        

type PtxInstructions(intent) = 
    let mutable instructions = []
    let spaces = String.replicate intent " "
    let ops = 
        Map [
            "add", fun expr1 expr2 -> <@@ %%expr1 + %%expr2 @@>
            "sub", fun expr1 expr2 -> <@@ %%expr1 - %%expr2 @@>
            "mul", fun expr1 expr2 -> <@@ %%expr1 * %%expr2 @@>
            "div", fun expr1 expr2 -> <@@ %%expr1 / %%expr2 @@>
            "ret", fun expr1 expr2 -> <@@ %%expr1 % %%expr2 @@>
            "eq", fun expr1 expr2 -> <@@ %%expr1 = %%expr2 @@>
            "ne", fun expr1 expr2 -> <@@ %%expr1 <> %%expr2 @@>
            "lt", fun expr1 expr2 -> <@@ %%expr1 < %%expr2 @@>
            "le", fun expr1 expr2 -> <@@ %%expr1 <= %%expr2 @@>
            "gt", fun expr1 expr2 -> <@@ %%expr1 > %%expr2 @@>
            "ge", fun expr1 expr2 -> <@@ %%expr1 >= %%expr2 @@>
        ]
    member x.New(instruction: string) = 
        instructions <- instruction::instructions
    override x.ToString() =
        let rec processInstructions acc (instructions: string list) = 
            match instructions with
            | [] -> acc
            | i1::i2::instructions when i2.EndsWith(":") ->
                processInstructions acc ((sprintf "%s %s" i2 i1)::instructions)
            | i::instructions when i.StartsWith("@") || i.StartsWith("L_") -> processInstructions (i::acc) instructions                
            | i::instructions -> processInstructions ((sprintf "%s%s" spaces i)::acc) instructions  
        String.Join(Environment.NewLine, processInstructions [] ("ret;"::instructions))
    member x.Eval(ptxOp, (v1, t1), (v2, t2)) = 
        let res = (ops.[ptxOp] (Expr.Value(v1, t1)) (Expr.Value(v2, t2))).EvalUntyped()
        res, res.GetType()

let rec compile (expr: Expr) =     
    let signature = PtxSignature()
    let declarations = PtxDeclarations(6)
    let instructions = PtxInstructions(6)
    let rec getFuncParams (expr: Expr) = 
        match expr with
        | Lambda(maybeTuple, expr) when FSharpType.IsTuple(maybeTuple.Type) -> //2 and more parameters
            let rec selectParams(i, data, expr) = 
                match expr with
                | Let(v, TupleGet(Var(t), j), expr') when t = maybeTuple && j = i -> 
                    let param = signature.New(v)
                    let reg = declarations.NewBinded(param)
                    sprintf "ld.param%s %s, [%s]" (reg.ptxType.ToString()) reg.Name param.name
                    |> instructions.New
                    selectParams(i+1, data, expr')
                | _ -> data, expr
            let data, funcBody = selectParams(0, signature, expr)
            getFuncParams funcBody
        | Lambda(unitParameter, funcBody) when unitParameter.Type = typeof<unit> -> // 0 parameters
            getFuncParams funcBody
        | Lambda(singleParameter, funcBody) ->
            let param = signature.New(singleParameter)
            let reg = declarations.NewBinded(param)
            sprintf "ld.param%s %s, [%s]" (reg.ptxType.ToString()) reg.Name param.name
            |> instructions.New
            getFuncParams funcBody
        | _ -> compileBody expr 
    and compileBody expr =       
        let operandWithOperand (opStr, modifier) expr1 expr2 = 
            match compileBody expr1, compileBody expr2 with
            | Choice1Of3 reg1, Choice1Of3 reg2 ->
                let resReg = declarations.FreeAndAlloc(reg1.ptxType, reg1, reg2)
                sprintf 
                    "%s%s%s %s, %s, %s;" 
                    opStr modifier (resReg.ptxType.ToString()) resReg.Name reg1.Name reg2.Name
                |> instructions.New
                Choice1Of3 resReg
            | Choice1Of3 reg1, Choice2Of3 (v, t) ->
                let resReg = declarations.FreeAndAlloc(reg1.ptxType, reg1)
                sprintf 
                    "%s%s%s %s, %s, %A;" 
                    opStr modifier (resReg.ptxType.ToString()) resReg.Name reg1.Name v
                |> instructions.New
                Choice1Of3 resReg
            | Choice2Of3 (v, t), Choice1Of3 reg1 ->
                let resReg = declarations.FreeAndAlloc(reg1.ptxType, reg1)
                sprintf 
                    "%s%s%s %s, %A, %s;" 
                    opStr modifier (resReg.ptxType.ToString()) resReg.Name v reg1.Name
                |> instructions.New
                Choice1Of3 resReg
            | Choice2Of3 (v1, t1), Choice2Of3 (v2, t2) ->                
                instructions.Eval(opStr, (v1, t1), (v2, t2))
                |> Choice2Of3
            | Choice3Of3 (), _ | _, Choice3Of3() -> failwith "Operand could not be null"
        let logicalOp (opStr, modifier) expr1 expr2 = 
            match compileBody expr1, compileBody expr2 with
            | Choice1Of3 reg1, Choice1Of3 reg2 ->                
                let predReg = declarations.FreeAndAlloc(PtxType.Pred, reg1, reg2)
                sprintf 
                    "setp.%s%s%s %s, %s, %s;" 
                    opStr modifier (reg1.ptxType.ToString()) predReg.Name reg1.Name reg2.Name
                |> instructions.New
                Choice1Of3 predReg
            | Choice1Of3 reg1, Choice2Of3 (v, t) ->
                let predReg = declarations.FreeAndAlloc(PtxType.Pred, reg1)
                sprintf 
                    "setp.%s%s%s %s, %s, %A;" 
                    opStr modifier (reg1.ptxType.ToString()) predReg.Name reg1.Name v
                |> instructions.New
                Choice1Of3 predReg
            | Choice2Of3 (v, t), Choice1Of3 reg1 ->
                let predReg = declarations.FreeAndAlloc(PtxType.Pred, reg1)
                sprintf 
                    "setp.%s%s%s %s, %A, %s;" 
                    opStr modifier (reg1.ptxType.ToString()) predReg.Name v reg1.Name
                |> instructions.New
                Choice1Of3 predReg
            | Choice2Of3 (v1, t1), Choice2Of3 (v2, t2) ->                
                instructions.Eval(opStr, (v1, t1), (v2, t2))
                |> Choice2Of3
            | Choice3Of3 (), _ | _, Choice3Of3() -> failwith "Operand could not be null"
        match expr with
        | Sequential (firstExpr, secondExpr) -> 
            match compileBody firstExpr with
            | Choice1Of3 reg -> declarations.Free(reg)
            | _ -> ()
            compileBody secondExpr
        | Let (v, expr, inExpr) ->
            let reg = declarations.NewBinded(v)
            match compileBody expr with
            | Choice1Of3 reg2 -> 
                declarations.Free(reg2)
                sprintf "mov%s %s, %s;" (reg.ptxType.ToString()) reg.Name reg2.Name
                |> instructions.New
                let res = compileBody inExpr
                declarations.Unbind(reg)
                res
            | Choice2Of3 (value, t) ->
                sprintf "mov%s %s, %A;" (reg.ptxType.ToString()) reg.Name value
                |> instructions.New
                let res = compileBody inExpr
                declarations.Unbind(reg)
                res
            | Choice3Of3 () -> compileBody inExpr
        | VarSet(v, expr) ->
            let bindedReg = declarations.NewBinded(v)
            match compileBody expr with
            | Choice1Of3 reg ->
                declarations.Free(reg)
                sprintf "mov%s %s, %s;" (bindedReg.ptxType.ToString()) bindedReg.Name reg.Name
                |> instructions.New                
                Choice3Of3()
            | Choice2Of3(value, t) ->
                sprintf "mov%s %s, %A;" (bindedReg.ptxType.ToString()) bindedReg.Name value
                |> instructions.New
                Choice3Of3()                
            | _ -> Choice3Of3()
        | Value(null, _) -> Choice3Of3()
        | Value(v:obj, t) -> Choice2Of3(v, t)
        | Var(v) -> 
            let bindedReg = declarations.NewBinded(v)
            Choice1Of3 bindedReg
        //a*b + c
        | SpecificCall <@@ (+) @@> (_, _, [SpecificCall <@@ (*) @@> (_, _, [aExpr; bExpr]); cExpr]) -> 
            match compileBody aExpr, compileBody bExpr, compileBody cExpr with
            | Choice1Of3 aReg, Choice1Of3 bReg, Choice1Of3 cReg ->
                let resReg = declarations.FreeAndAlloc(aReg.ptxType, aReg, bReg, cReg)
                sprintf "mad.lo%s %s, %s, %s, %s;"
                    (resReg.ptxType.ToString()) resReg.Name aReg.Name bReg.Name cReg.Name
                |> instructions.New
                Choice1Of3 resReg
            | Choice1Of3 aReg, Choice1Of3 bReg, Choice2Of3(v, t) ->
                let resReg = declarations.FreeAndAlloc(aReg.ptxType, aReg, bReg)
                sprintf "mad.lo%s %s, %s, %s, %A;"
                    (resReg.ptxType.ToString()) resReg.Name aReg.Name bReg.Name v
                |> instructions.New
                Choice1Of3 resReg  
            | Choice1Of3 aReg,  Choice2Of3(v, t), Choice1Of3 cReg ->
                let resReg = declarations.FreeAndAlloc(aReg.ptxType, aReg, cReg)
                sprintf "mad.lo%s %s, %s, %A, %s;"
                    (resReg.ptxType.ToString()) resReg.Name aReg.Name v cReg.Name
                |> instructions.New
                Choice1Of3 resReg        
            | Choice2Of3(v, t), Choice1Of3 bReg, Choice1Of3 cReg ->
                let resReg = declarations.FreeAndAlloc(bReg.ptxType, bReg, cReg)
                sprintf "mad.lo%s %s, %A, %s, %s;"
                    (resReg.ptxType.ToString()) resReg.Name v bReg.Name cReg.Name
                |> instructions.New
                Choice1Of3 resReg 
            | Choice2Of3(v1, t1), Choice1Of3 bReg, Choice2Of3 (v2, t2) ->
                let resReg = declarations.FreeAndAlloc(bReg.ptxType, bReg)
                sprintf "mad.lo%s %s, %A, %s, %A;"
                    (resReg.ptxType.ToString()) resReg.Name v1 bReg.Name v2
                |> instructions.New
                Choice1Of3 resReg 
            | Choice1Of3 aReg, Choice2Of3(v1, t1), Choice2Of3 (v2, t2) ->
                let resReg = declarations.FreeAndAlloc(aReg.ptxType, aReg)
                sprintf "mad.lo%s %s, %s, %A, %A;"
                    (resReg.ptxType.ToString()) resReg.Name aReg.Name v1 v2
                |> instructions.New
                Choice1Of3 resReg
            | Choice2Of3(v1, t1), Choice2Of3(v2, t2), Choice1Of3 cReg ->
                let resReg = declarations.FreeAndAlloc(cReg.ptxType, cReg)
                let v, _ = instructions.Eval("mul", (v1, t1), (v2, t2))
                sprintf "add.lo%s %s, %A, %s;"
                    (resReg.ptxType.ToString()) resReg.Name v cReg.Name
                |> instructions.New
                Choice1Of3 resReg
            | Choice2Of3(v1, t1), Choice2Of3(v2, t2), Choice2Of3(v3, t3) ->
                let v, t = instructions.Eval("mul", (v1, t1), (v2, t2))
                instructions.Eval("add", (v, t), (v3, t3)) |> Choice2Of3
            | Choice3Of3(), _, _ | _, _, Choice3Of3() | _, Choice3Of3(), _ ->
                failwith "mad ptx could not be build from %A, %A, %A" aExpr bExpr cExpr
        | SpecificCall <@@ (+) @@> (_, _, [expr1; expr2]) -> operandWithOperand ("add", ".lo") expr1 expr2
        | SpecificCall <@@ (-) @@> (_, _, [expr1; expr2]) -> operandWithOperand ("sub", "") expr1 expr2
        | SpecificCall <@@ (*) @@> (_, _, [expr1; expr2]) -> operandWithOperand ("mul", "") expr1 expr2
        | SpecificCall <@@ (/) @@> (_, _, [expr1; expr2]) -> operandWithOperand ("div", "") expr1 expr2
        | SpecificCall <@@ (%) @@> (_, _, [expr1; expr2]) -> operandWithOperand ("rem", "") expr1 expr2
        | SpecificCall <@@ (>) @@> (_, _, [expr1; expr2]) -> logicalOp ("gt", "") expr1 expr2
        | SpecificCall <@@ (>=) @@> (_, _, [expr1; expr2]) -> logicalOp ("ge", "") expr1 expr2
        | SpecificCall <@@ (<) @@> (_, _, [expr1; expr2]) -> logicalOp ("lt", "") expr1 expr2
        | SpecificCall <@@ (<=) @@> (_, _, [expr1; expr2]) -> logicalOp ("le", "") expr1 expr2
        | SpecificCall <@@ (=) @@> (_, _, [expr1; expr2]) -> logicalOp ("eq", "") expr1 expr2
        | SpecificCall <@@ (<>) @@> (_, _, [expr1; expr2]) -> logicalOp ("ne", "") expr1 expr2
        | ForIntegerRangeLoop (loopVar, startRangeExpr, endRangeExpr, bodyExpr) ->
            let loopReg = declarations.NewBinded(loopVar)
            match compileBody startRangeExpr, compileBody endRangeExpr with
            | Choice1Of3 reg1, Choice1Of3 reg2 ->
                sprintf "mov%s %s, %s;" (loopReg.ptxType.ToString()) loopReg.Name reg1.Name
                |> instructions.New
                let predReg = declarations.NewReg(PtxType.Pred)
                let label1 = declarations.NewLabel()
                sprintf "%s:" label1 |> instructions.New
                sprintf "setp.ge%s %s, %s, %s;" (loopReg.ptxType.ToString()) predReg.Name loopReg.Name reg2.Name
                |> instructions.New
                let label2 = declarations.NewLabel()
                sprintf "@%s bra %s;" predReg.Name label2 |> instructions.New 
                declarations.NewBindingSpace()
                match compileBody bodyExpr with
                | Choice1Of3 reg -> declarations.Free(reg)
                | _ -> ()
                declarations.UnbindSpace()
                sprintf "add.lo%s %s, %s, 1;" (loopReg.ptxType.ToString()) loopReg.Name loopReg.Name
                |> instructions.New 
                sprintf "bra %s;" label1 |> instructions.New 
                sprintf "%s:" label2 |> instructions.New
                declarations.Free(reg1, reg2, predReg)
            | Choice1Of3 reg1, Choice2Of3 (v, t) ->
                sprintf "mov%s %s, %s;" (loopReg.ptxType.ToString()) loopReg.Name reg1.Name
                |> instructions.New
                let predReg = declarations.NewReg(PtxType.Pred)
                let label1 = declarations.NewLabel()
                sprintf "%s:" label1 |> instructions.New
                sprintf "setp.ge%s %s, %s, %A;" (loopReg.ptxType.ToString()) predReg.Name loopReg.Name v
                |> instructions.New
                let label2 = declarations.NewLabel()
                sprintf "@%s bra %s;" predReg.Name label2 |> instructions.New 
                declarations.NewBindingSpace()
                match compileBody bodyExpr with
                | Choice1Of3 reg -> declarations.Free(reg)
                | _ -> ()
                declarations.UnbindSpace()
                sprintf "add.lo%s %s, %s, 1;" (loopReg.ptxType.ToString()) loopReg.Name loopReg.Name
                |> instructions.New 
                sprintf "bra %s;" label1 |> instructions.New 
                sprintf "%s:" label2 |> instructions.New
                declarations.Free(reg1, predReg)    
            | Choice2Of3 (v, t), Choice1Of3 reg2 ->
                sprintf "mov%s %s, %A;" (loopReg.ptxType.ToString()) loopReg.Name v
                |> instructions.New
                let predReg = declarations.NewReg(PtxType.Pred)
                let label1 = declarations.NewLabel()
                sprintf "%s:" label1 |> instructions.New
                sprintf "setp.ge%s %s, %s, %A;" (loopReg.ptxType.ToString()) predReg.Name loopReg.Name v
                |> instructions.New
                let label2 = declarations.NewLabel()
                sprintf "@%s bra %s;" predReg.Name label2 |> instructions.New 
                declarations.NewBindingSpace()
                match compileBody bodyExpr with
                | Choice1Of3 reg -> declarations.Free(reg)
                | _ -> ()
                declarations.UnbindSpace()
                sprintf "add.lo%s %s, %s, 1;" (loopReg.ptxType.ToString()) loopReg.Name loopReg.Name
                |> instructions.New 
                sprintf "bra %s;" label1 |> instructions.New 
                sprintf "%s:" label2 |> instructions.New
                declarations.Free(reg2, predReg)    
            | Choice2Of3 (v1, t1), Choice2Of3 (v2, t2) ->
                sprintf "mov%s %s, %A;" (loopReg.ptxType.ToString()) loopReg.Name v1
                |> instructions.New
                let predReg = declarations.NewReg(PtxType.Pred)
                let label1 = declarations.NewLabel()
                sprintf "%s:" label1 |> instructions.New
                sprintf "setp.ge%s %s, %s, %A;" (loopReg.ptxType.ToString()) predReg.Name loopReg.Name v2
                |> instructions.New
                let label2 = declarations.NewLabel()
                sprintf "@%s bra %s;" predReg.Name label2 |> instructions.New 
                declarations.NewBindingSpace()
                match compileBody bodyExpr with
                | Choice1Of3 reg -> declarations.Free(reg)
                | _ -> ()
                declarations.UnbindSpace()
                sprintf "add.lo%s %s, %s, 1;" (loopReg.ptxType.ToString()) loopReg.Name loopReg.Name
                |> instructions.New 
                sprintf "bra %s;" label1 |> instructions.New 
                sprintf "%s:" label2 |> instructions.New
                declarations.Free(predReg)  
            | _, _ -> failwith "range start or end was not computed: %A, %A" startRangeExpr endRangeExpr
            declarations.Unbind(loopReg)
            Choice3Of3()
        | WhileLoop (condExpr, bodyExpr) ->
            let label1 = declarations.NewLabel()
            sprintf "%s:" label1 |> instructions.New
            match compileBody condExpr with
            | Choice1Of3 predReg ->
                let label2 = declarations.NewLabel()
                sprintf "@!%s bra %s;" predReg.Name label2 |> instructions.New   
                declarations.NewBindingSpace()                             
                match compileBody bodyExpr with
                | Choice1Of3 reg -> declarations.Free(reg)
                | _ -> ()
                declarations.UnbindSpace()
                sprintf "bra %s;" label1 |> instructions.New
                sprintf "%s:" label2 |> instructions.New
                declarations.Free(predReg)
                Choice3Of3()
            | _ -> Choice3Of3 ()
//        | IfThenElse(condExpr, Value(v1, t1), Value(v2, t2)) ->
//            let cond = condExpr.EvalUntyped()
//            if (Convert.ToBoolean(cond)) then Choice2Of3(v1, t1) else Choice2Of3(v2, t2)
        | IfThenElse(condExpr, ifExpr, elseExpr) ->
            match compileBody condExpr with
            | Choice1Of3 predReg ->
                let label = declarations.NewLabel()
                sprintf "@!%s bra %s;" predReg.Name label
                |> instructions.New
                declarations.Free(predReg)
                let ifReg = compileBody ifExpr
                let returnRegOpt = 
                    match ifReg with
                    | Choice1Of3 reg -> 
                        let outReg = declarations.FreeAndAlloc(reg.ptxType, reg)
                        sprintf "mov%s %s, %s;" (outReg.ptxType.ToString()) outReg.Name reg.Name |> instructions.New
                        Some outReg
                    | Choice2Of3(v, t) ->
                        let outReg = declarations.NewReg(t.ToPtxType())
                        sprintf "mov%s %s, %A;" (outReg.ptxType.ToString()) outReg.Name v |> instructions.New
                        Some outReg
                    | _ -> None
                let label2 = declarations.NewLabel()
                sprintf "bra %s;" label2 |> instructions.New
                sprintf "%s:" label
                |> instructions.New
                let elseReg = compileBody elseExpr                
                let out = 
                    match elseReg, returnRegOpt with
                    | Choice1Of3 reg, Some outReg -> 
                        declarations.Free(reg)
                        sprintf "mov%s %s, %s;" (outReg.ptxType.ToString()) outReg.Name reg.Name |> instructions.New                        
                        Choice1Of3 outReg                        
                    | Choice2Of3(v,t), Some outReg ->
                        sprintf "mov%s %s, %A;" (outReg.ptxType.ToString()) outReg.Name v |> instructions.New
                        Choice1Of3 outReg
                    | _, _ -> Choice3Of3()                        
                sprintf "%s:" label2 |> instructions.New
                out
            | Choice2Of3 (v, t) ->
                if Convert.ToBoolean(v) then compileBody ifExpr
                else compileBody elseExpr
            | Choice3Of3 () -> compileBody elseExpr
        | Call(None, methodInfo, [arrayVarExpr; indexExpr ]) when methodInfo.Name = "GetArray" ->
            match compileBody indexExpr, compileBody arrayVarExpr with
            | Choice2Of3 (:? int as index, _), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                let valueReg = declarations.NewReg(param.ptxType)
                if index = 0 then
                    sprintf "ld%s%s %s, [%s];" 
                        (memoryType.ToString()) (valueReg.ptxType.ToString()) valueReg.Name reg.Name
                else
                    sprintf "ld%s%s %s, [%s + %d];" 
                        (memoryType.ToString()) (valueReg.ptxType.ToString()) valueReg.Name reg.Name (index*valueReg.ptxType.ByteSize)
                |> instructions.New
                Choice1Of3 valueReg
            | Choice2Of3 (v, _), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                failwithf "Value %A is bad for index" v
            | Choice3Of3 (), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                failwith "Null value is bad for index"
            | Choice1Of3 indexReg, Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                let newIndexReg = declarations.FreeAndAlloc(indexReg.ptxType, indexReg)
                sprintf "mad.lo%s %s, %s, %d, %s;"
                    (newIndexReg.ptxType.ToString()) newIndexReg.Name indexReg.Name param.ptxType.ByteSize reg.Name
                |> instructions.New
                let valueReg = declarations.NewReg(param.ptxType)
                sprintf "ld%s%s %s, [%s];" 
                    (memoryType.ToString()) (valueReg.ptxType.ToString()) valueReg.Name newIndexReg.Name
                |> instructions.New
                declarations.Free(newIndexReg)
                Choice1Of3 valueReg
            | _, Choice1Of3 reg -> failwithf "Register does not binded to parameter %A" reg
            | _, Choice2Of3 (v, _) -> failwithf "Array variable %A is unsupported" v
            | _, Choice3Of3 () -> failwith "Array var could not be detected" 
        | Call(None, methodInfo, [arrayVarExpr; index1Expr; index2Expr ]) when methodInfo.Name = "GetArray2D" ->
            match compileBody index1Expr, compileBody index2Expr, compileBody arrayVarExpr with
            | _, Choice3Of3 (), _ | Choice3Of3(), _, _ -> 
                failwith "Null value is bad for index"
            | Choice2Of3 (:? int as index1, _), Choice2Of3 (:? int as index2, _), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                let valueReg = declarations.NewReg(param.ptxType)
                match index1, index2 with
                | 0, 0 ->
                    sprintf "ld%s%s %s, [%s];" 
                        (memoryType.ToString()) (valueReg.ptxType.ToString()) valueReg.Name reg.Name
                | 0, _ ->
                    sprintf "ld%s%s %s, [%s + %d];" 
                        (memoryType.ToString()) (valueReg.ptxType.ToString()) valueReg.Name reg.Name (index2*valueReg.ptxType.ByteSize)
                | _, _ ->
                    let length2Reg = reg.props.["Length2"] 
                    let indexReg = declarations.NewReg(length2Reg.ptxType)                    
                    sprintf "mad.lo%s %s, %d, %s, %d;" (length2Reg.ptxType.ToString()) indexReg.Name index1 length2Reg.Name index2
                    |> instructions.New
                    sprintf "mad.lo%s %s, %d, %s, %s;" (indexReg.ptxType.ToString()) indexReg.Name valueReg.ptxType.ByteSize indexReg.Name reg.Name
                    |> instructions.New
                    declarations.Free(indexReg)
                    sprintf "ld%s%s %s, [%s];" 
                        (memoryType.ToString()) (valueReg.ptxType.ToString()) valueReg.Name indexReg.Name                                        
                |> instructions.New
                Choice1Of3 valueReg
            | Choice2Of3(v1, _), Choice2Of3 (v2, _), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                failwithf "Values %A, %A are bad for index" v1 v2
            | Choice2Of3 (:? int as index1, _), Choice1Of3 indexReg2, Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                let newIndexReg = declarations.FreeAndAlloc(indexReg2.ptxType, indexReg2)
                let length2Reg = reg.props.["Length2"] 
                sprintf "mad.lo%s %s, %s, %d, %s;"
                    (newIndexReg.ptxType.ToString()) newIndexReg.Name length2Reg.Name index1 indexReg2.Name
                |> instructions.New
                sprintf "mad.lo%s %s, %s, %d, %s;"
                    (newIndexReg.ptxType.ToString()) newIndexReg.Name newIndexReg.Name param.ptxType.ByteSize reg.Name
                |> instructions.New
                let valueReg = declarations.NewReg(param.ptxType)
                sprintf "ld%s%s %s, [%s];" 
                    (memoryType.ToString()) (valueReg.ptxType.ToString()) valueReg.Name newIndexReg.Name
                |> instructions.New
                declarations.Free(newIndexReg)
                Choice1Of3 valueReg
            | Choice1Of3 indexReg1, Choice1Of3 indexReg2, Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                let newIndexReg = declarations.FreeAndAlloc(indexReg2.ptxType, indexReg2, indexReg1)
                let length2Reg = reg.props.["Length2"] 
                sprintf "mad.lo%s %s, %s, %s, %s;"
                    (newIndexReg.ptxType.ToString()) newIndexReg.Name length2Reg.Name indexReg1.Name indexReg2.Name
                |> instructions.New
                sprintf "mad.lo%s %s, %s, %d, %s;"
                    (newIndexReg.ptxType.ToString()) newIndexReg.Name newIndexReg.Name param.ptxType.ByteSize reg.Name
                |> instructions.New
                let valueReg = declarations.NewReg(param.ptxType)
                sprintf "ld%s%s %s, [%s];" 
                    (memoryType.ToString()) (valueReg.ptxType.ToString()) valueReg.Name newIndexReg.Name
                |> instructions.New
                declarations.Free(newIndexReg)
                Choice1Of3 valueReg
            | Choice1Of3 indexReg1, Choice2Of3 (:? int as index2, _), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                let newIndexReg = declarations.FreeAndAlloc(indexReg1.ptxType, indexReg1)
                let length2Reg = reg.props.["Length2"] 
                sprintf "mad.lo%s %s, %s, %s, %d;"
                    (newIndexReg.ptxType.ToString()) newIndexReg.Name length2Reg.Name indexReg1.Name index2
                |> instructions.New
                sprintf "mad.lo%s %s, %s, %d, %s;"
                    (newIndexReg.ptxType.ToString()) newIndexReg.Name newIndexReg.Name param.ptxType.ByteSize reg.Name
                |> instructions.New
                let valueReg = declarations.NewReg(param.ptxType)
                sprintf "ld%s%s %s, [%s];" 
                    (memoryType.ToString()) (valueReg.ptxType.ToString()) valueReg.Name newIndexReg.Name
                |> instructions.New
                declarations.Free(newIndexReg)
                Choice1Of3 valueReg
            | _, _, Choice1Of3 reg -> failwithf "Register does not binded to parameter %A" reg
            | _, _, Choice2Of3 (v, _) -> failwithf "Array variable %A is unsupported" v
            | _, _, Choice3Of3 () -> failwith "Array var could not be detected"                           
        | Call(None, methodInfo, [arrayVarExpr; indexExpr; setValueExpr ]) when methodInfo.Name = "SetArray" ->
            match compileBody setValueExpr, compileBody indexExpr, compileBody arrayVarExpr with
            | Choice2Of3 (v, t), Choice2Of3 (:? int as index, _), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                if index = 0 then
                    sprintf "st%s%s [%s], %A;" 
                        (memoryType.ToString()) (param.ptxType.ToString()) reg.Name v
                else
                    sprintf "st%s%s [%s + %d], %A;" 
                        (memoryType.ToString()) (param.ptxType.ToString()) reg.Name (index*param.ptxType.ByteSize) v
                |> instructions.New
                Choice3Of3()
            | Choice1Of3 resReg, Choice2Of3 (:? int as index, _), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                declarations.Free(resReg)
                if index = 0 then
                    sprintf "st%s%s [%s], %s;" 
                        (memoryType.ToString()) (param.ptxType.ToString()) reg.Name resReg.Name
                else
                    sprintf "st%s%s [%s + %d], %s;" 
                        (memoryType.ToString()) (param.ptxType.ToString()) reg.Name (index*param.ptxType.ByteSize) resReg.Name
                |> instructions.New                
                Choice3Of3()
            | Choice3Of3 (), Choice2Of3 (:? int as index, _), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                Choice3Of3()
            | _, Choice2Of3 (v, _), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                failwithf "Value %A is bad for index" v
            | _, Choice3Of3 (), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                failwith "Null value is bad for index"
            | Choice2Of3 (v, t), Choice1Of3 indexReg, Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                let newIndexReg = declarations.FreeAndAlloc(indexReg.ptxType, indexReg)
                sprintf "mad.lo%s %s, %s, %d, %s;"
                    (newIndexReg.ptxType.ToString()) newIndexReg.Name indexReg.Name param.ptxType.ByteSize reg.Name
                |> instructions.New
                sprintf "st%s%s [%s], %A;" 
                    (memoryType.ToString()) (param.ptxType.ToString()) newIndexReg.Name v
                |> instructions.New
                declarations.Free(newIndexReg)
                Choice3Of3()
            | Choice1Of3 resReg, Choice1Of3 indexReg, Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                let newIndexReg = declarations.FreeAndAlloc(indexReg.ptxType, indexReg)
                sprintf "mad.lo%s %s, %s, %d, %s;"
                    (newIndexReg.ptxType.ToString()) newIndexReg.Name indexReg.Name param.ptxType.ByteSize reg.Name
                |> instructions.New
                sprintf "st%s%s [%s], %s;" 
                    (memoryType.ToString()) (param.ptxType.ToString()) newIndexReg.Name resReg.Name
                |> instructions.New
                declarations.Free(newIndexReg, resReg)
                Choice3Of3()
            | Choice3Of3 (), Choice1Of3 indexReg, Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) ->  
                declarations.Free(indexReg)                
                Choice3Of3()
            | _, _, Choice1Of3 reg -> failwithf "Register does not binded to parameter %A" reg
            | _, _, Choice2Of3 (v, _) -> failwithf "Array variable %A is unsupported" v
            | _, _, Choice3Of3 () -> failwith "Array var could not be detected"
        | Call(None, methodInfo, [arrayVarExpr; index1Expr; index2Expr; setValueExpr ]) when methodInfo.Name = "SetArray2D" ->
            match compileBody setValueExpr, compileBody index1Expr, compileBody index2Expr, compileBody arrayVarExpr with
            | Choice2Of3 (v, t), Choice2Of3 (:? int as index1, _), Choice2Of3 (:? int as index2, _), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                match index1, index2 with
                | 0, 0 ->
                    sprintf "st%s%s [%s], %A;" 
                        (memoryType.ToString()) (param.ptxType.ToString()) reg.Name v
                | 0, _ ->
                    sprintf "st%s%s [%s + %d], %A;" 
                        (memoryType.ToString()) (param.ptxType.ToString()) reg.Name (index2*param.ptxType.ByteSize) v
                | _, _ ->
                    let length2Reg = reg.props.["Length2"] 
                    let indexReg = declarations.NewReg(length2Reg.ptxType)
                    sprintf "mad.lo%s %s, %d, %s, %d;" (indexReg.ptxType.ToString()) indexReg.Name index1 length2Reg.Name index2
                    |> instructions.New
                    sprintf "mad.lo%s %s, %s, %d, %s;" (indexReg.ptxType.ToString()) indexReg.Name indexReg.Name param.ptxType.ByteSize reg.Name
                    |> instructions.New
                    declarations.Free(indexReg)
                    sprintf "st%s%s [%s], %A;" 
                        (memoryType.ToString()) (param.ptxType.ToString()) indexReg.Name v                
                |> instructions.New
                Choice3Of3()
            | Choice1Of3 resReg, Choice2Of3 (:? int as index1, _), Choice2Of3 (:? int as index2, _), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                declarations.Free(resReg)
                match index1, index2 with
                | 0, 0 ->
                    sprintf "st%s%s [%s], %s;" 
                        (memoryType.ToString()) (param.ptxType.ToString()) reg.Name resReg.Name
                | 0, _ ->
                    sprintf "st%s%s [%s + %d], %s;" 
                        (memoryType.ToString()) (param.ptxType.ToString()) reg.Name (index2*param.ptxType.ByteSize) resReg.Name
                | _, _ ->
                    let length2Reg = reg.props.["Length2"] 
                    let indexReg = declarations.NewReg(length2Reg.ptxType)
                    sprintf "mad.lo%s %s, %d, %s, %d;" (indexReg.ptxType.ToString()) indexReg.Name index1 length2Reg.Name index2
                    |> instructions.New
                    sprintf "mad.lo%s %s, %s, %d, %s;" (indexReg.ptxType.ToString()) indexReg.Name indexReg.Name param.ptxType.ByteSize reg.Name
                    |> instructions.New
                    declarations.Free(indexReg)
                    sprintf "st%s%s [%s], %s;" 
                        (memoryType.ToString()) (param.ptxType.ToString()) indexReg.Name resReg.Name              
                |> instructions.New
                Choice3Of3()
            | _, Choice3Of3 (), _, Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) ->
                failwith "Null value is bad for index"
            | _, _, Choice3Of3 (), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) ->
                failwith "Null value is bad for index"
            | Choice2Of3 (v, t), Choice1Of3 indexReg1, Choice1Of3 indexReg2, Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                let newIndexReg1 = declarations.FreeAndAlloc(indexReg1.ptxType, indexReg1)
                sprintf "mad.lo%s %s, %s, %s, %s;" //TODO - ptx type conversion
                    (newIndexReg1.ptxType.ToString()) newIndexReg1.Name indexReg1.Name reg.props.["Length1"].Name indexReg2.Name//TODO: check
                |> instructions.New
                let newIndexReg2 = declarations.FreeAndAlloc(indexReg2.ptxType, indexReg2)
                sprintf "mad.lo%s %s, %s, %d, %s;"
                    (newIndexReg2.ptxType.ToString()) newIndexReg2.Name newIndexReg1.Name param.ptxType.ByteSize reg.Name
                |> instructions.New
                sprintf "st%s%s [%s], %A;" 
                    (memoryType.ToString()) (param.ptxType.ToString()) newIndexReg2.Name v
                |> instructions.New
                declarations.Free(newIndexReg1, newIndexReg2)
                Choice3Of3()
            | Choice2Of3 (v, t), Choice1Of3 indexReg1, Choice2Of3(:? int as index2, _), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                let newIndexReg1 = declarations.FreeAndAlloc(indexReg1.ptxType, indexReg1)
                sprintf "mad.lo%s %s, %s, %s, %d;" //TODO - ptx type conversion
                    (newIndexReg1.ptxType.ToString()) newIndexReg1.Name indexReg1.Name reg.props.["Length1"].Name index2//TODO: check
                |> instructions.New
                let newIndexReg2 = declarations.NewReg(indexReg1.ptxType)
                sprintf "mad.lo%s %s, %s, %d, %s;"
                    (newIndexReg2.ptxType.ToString()) newIndexReg2.Name newIndexReg1.Name param.ptxType.ByteSize reg.Name
                |> instructions.New
                sprintf "st%s%s [%s], %A;" 
                    (memoryType.ToString()) (param.ptxType.ToString()) newIndexReg2.Name v
                |> instructions.New
                declarations.Free(newIndexReg1, newIndexReg2)
                Choice3Of3()
            | Choice2Of3 (v, t), Choice2Of3(:? int as index1, _), Choice1Of3 indexReg2, Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> 
                let newIndexReg1 = declarations.NewReg(indexReg2.ptxType)
                sprintf "mad.lo%s %s, %d, %s, %s;" //TODO - ptx type conversion
                    (newIndexReg1.ptxType.ToString()) newIndexReg1.Name index1 reg.props.["Length1"].Name indexReg2.Name//TODO: check
                |> instructions.New
                let newIndexReg2 = declarations.FreeAndAlloc(indexReg2.ptxType, indexReg2)
                sprintf "mad.lo%s %s, %s, %d, %s;"
                    (newIndexReg2.ptxType.ToString()) newIndexReg2.Name newIndexReg1.Name param.ptxType.ByteSize reg.Name
                |> instructions.New
                sprintf "st%s%s [%s], %A;" 
                    (memoryType.ToString()) (param.ptxType.ToString()) newIndexReg2.Name v
                |> instructions.New
                declarations.Free(newIndexReg1, newIndexReg2)
                Choice3Of3()
            | Choice1Of3 resReg, Choice1Of3 indexReg1, Choice1Of3 indexReg2, Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> //TODO last match
                let newIndexReg1 = declarations.FreeAndAlloc(indexReg1.ptxType, indexReg1)
                sprintf "mad.lo%s %s, %s, %s, %s;" //TODO - ptx type conversion
                    (newIndexReg1.ptxType.ToString()) newIndexReg1.Name indexReg1.Name reg.props.["Length1"].Name indexReg2.Name//TODO: check
                |> instructions.New
                let newIndexReg2 = declarations.FreeAndAlloc(indexReg2.ptxType, indexReg2)
                sprintf "mad.lo%s %s, %s, %d, %s;"
                    (newIndexReg2.ptxType.ToString()) newIndexReg2.Name newIndexReg1.Name param.ptxType.ByteSize reg.Name
                |> instructions.New
                sprintf "st%s%s [%s], %s;" 
                    (memoryType.ToString()) (param.ptxType.ToString()) newIndexReg2.Name resReg.Name
                |> instructions.New
                declarations.Free(newIndexReg1, newIndexReg2, resReg)
                Choice3Of3()
            | Choice1Of3 resReg, Choice2Of3(:? int as index1, _), Choice1Of3 indexReg2, Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> //TODO last match
                let newIndexReg1 = declarations.NewReg(indexReg2.ptxType)
                sprintf "mad.lo%s %s, %d, %s, %s;" //TODO - ptx type conversion
                    (newIndexReg1.ptxType.ToString()) newIndexReg1.Name index1 reg.props.["Length1"].Name indexReg2.Name//TODO: check
                |> instructions.New
                let newIndexReg2 = declarations.FreeAndAlloc(indexReg2.ptxType, indexReg2)
                sprintf "mad.lo%s %s, %s, %d, %s;"
                    (newIndexReg2.ptxType.ToString()) newIndexReg2.Name newIndexReg1.Name param.ptxType.ByteSize reg.Name
                |> instructions.New
                sprintf "st%s%s [%s], %s;" 
                    (memoryType.ToString()) (param.ptxType.ToString()) newIndexReg2.Name resReg.Name
                |> instructions.New
                declarations.Free(newIndexReg1, newIndexReg2, resReg)
                Choice3Of3()
            | Choice1Of3 resReg, Choice1Of3 indexReg1, Choice2Of3(:? int as index2, _), Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> //TODO last match
                let newIndexReg1 = declarations.FreeAndAlloc(indexReg1.ptxType, indexReg1)
                sprintf "mad.lo%s %s, %s, %s, %d;" //TODO - ptx type conversion
                    (newIndexReg1.ptxType.ToString()) newIndexReg1.Name indexReg1.Name reg.props.["Length1"].Name index2//TODO: check
                |> instructions.New
                let newIndexReg2 = declarations.NewReg(indexReg1.ptxType)
                sprintf "mad.lo%s %s, %s, %d, %s;"
                    (newIndexReg2.ptxType.ToString()) newIndexReg2.Name newIndexReg1.Name param.ptxType.ByteSize reg.Name
                |> instructions.New
                sprintf "st%s%s [%s], %s;" 
                    (memoryType.ToString()) (param.ptxType.ToString()) newIndexReg2.Name resReg.Name
                |> instructions.New
                declarations.Free(newIndexReg1, newIndexReg2, resReg)
                Choice3Of3()
            | Choice3Of3 (), Choice1Of3 indexReg1, Choice1Of3 indexReg2, Choice1Of3 ({parameter = Some ({pointerTo = Some memoryType} as param)} as reg) -> //TODO last match
                declarations.Free(indexReg1, indexReg2)                
                Choice3Of3()
            | _, _, Choice2Of3 (v, _), _ | _, Choice2Of3 (v, _), _, _-> 
                failwithf "Value %A is bad for index" v
            | _, _, _, Choice1Of3 reg -> failwithf "Register does not binded to parameter %A" reg //TODO last match
            | _, _, _, Choice2Of3 (v, _) -> failwithf "Array variable %A is unsupported" v //TODO last match
            | _, _, _, Choice3Of3 () -> failwith "Array var could not be detected"                 //TODO last match
        | PropertyGet(Some varExpr, propInfo, []) ->
            match compileBody varExpr with
            | Choice1Of3 reg when reg.props.ContainsKey(propInfo.Name) -> 
                Choice1Of3 reg.props.[propInfo.Name]
            | Choice1Of3 reg -> failwithf "Register %A does not have property %A, %A" reg propInfo.Name expr
            | v -> failwithf "Could not get property %A from %A, %A" propInfo.Name v expr
        | PropertySet(Some varExpr, propInfo, [], assignedExpr) ->
            match compileBody varExpr with
            | Choice1Of3 reg when reg.props.ContainsKey(propInfo.Name) -> 
                let reg = reg.props.[propInfo.Name]
                match compileBody assignedExpr with
                | Choice1Of3 assignedReg -> 
                    sprintf "mov%s %s, %s;" (reg.ptxType.ToString()) reg.Name assignedReg.Name
                    |> instructions.New
                    Choice3Of3()
                | Choice2Of3(v, t) ->
                    sprintf "mov%s %s, %A;" (reg.ptxType.ToString()) reg.Name v
                    |> instructions.New
                    Choice3Of3()                     
                | v -> v
            | Choice1Of3 reg -> failwithf "Register %A does not have property %A, %A" reg propInfo.Name expr
            | v -> failwithf "Could not get property %A from %A, %A" propInfo.Name v expr            
        | v -> failwithf "Unsupported instruction %A" v
    getFuncParams expr |> ignore    
    signature, declarations, instructions    

let compileToString expr = 
    let signature, declarations, instructions = compile expr 
    sprintf "%s {%s%s%s%s%s}" 
        (signature.AsKernel("test")) 
        Environment.NewLine
        (declarations.ToString()) 
        Environment.NewLine
        (instructions.ToString())    
        Environment.NewLine