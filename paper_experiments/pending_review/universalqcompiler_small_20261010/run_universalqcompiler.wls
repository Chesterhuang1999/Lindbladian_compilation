(* Run with wolframscript -file this.wls vendor-dir input.json result.json. *)
{sourceDirectory, inputFile, outputFile} = Take[$ScriptCommandLine, -3];
PrependTo[$Path, sourceDirectory];
Get[FileNameJoin[{sourceDirectory, "QI.m"}]];
Get[FileNameJoin[{sourceDirectory, "UniversalQCompiler.m"}]];

data = Import[inputFile, "RawJSON"];
If[data["status"] =!= "prepared", Print["Input is not eligible for compilation"]; Exit[2]];
channel = Map[#[[1]] + I #[[2]] &, data["compiler_kraus"], {3}];
inputDimension = data["input_dimension"];
outputDimension = data["output_dimension"];
inputError = Norm[Total[ConjugateTranspose[#].# & /@ channel] - IdentityMatrix[inputDimension], "Frobenius"];
If[!NumericQ[inputError] || inputError > 10^-10, Print["Input is not trace preserving"]; Exit[3]];

(* Use the upstream channel constructor and its default isometry selection.
   Keep native gate labels and angles; conversion to Qiskit must reverse qubit
   numbering and negate rotation angles, as in the upstream QASM converter. *)
{elapsed, gates} = AbsoluteTiming[
  TimeConstrained[
    Catch[DecChannelInQCM[channel, TryToCompress -> True, DecomposeIso -> "DecIsometry"]],
    600, $Aborted
  ]
];
If[!ListQ[gates],
  Export[outputFile, <|"status" -> "synthesis_failed", "compile_time_s" -> elapsed,
    "details" -> ToString[gates, InputForm]|>, "RawJSON"]; Exit[4]
];
Put[gates, outputFile <> ".gates.wl"];
nativeCounts = Counts[First /@ gates];

(* Reconstruct the complete channel, including trace-out, as in upstream tests. *)
verified = TimeConstrained[
  Catch[
    outputChannel = CreateOperationFromGateList[NGateList[gates]];
    If[Length[Dimensions[outputChannel]] == 2, outputChannel = {outputChannel}];
    targetChoi = KrausToChoi[channel];
    Norm[N[KrausToChoi[outputChannel] - targetChoi], "Frobenius"] /
      Norm[N[targetChoi], "Frobenius"]
  ], 600, $Aborted
];
status = If[NumericQ[verified] && verified < 10^-8, "verified", "verification_failed"];
result = <|
  "status" -> status, "case" -> data["case"], "mode" -> data["mode"],
  "wolfram_version" -> $Version, "compile_time_s" -> elapsed,
  "input_tp_frobenius_error" -> inputError,
  "choi_relative_error" -> If[NumericQ[verified], verified, Null],
  "verification_details" -> ToString[verified, InputForm],
  "native_gate_type_counts" -> Association[KeyValueMap[ToString[#1] -> #2 &, nativeCounts]],
  "native_cnot_count" -> CNOTCount[gates],
  "native_gate_list_file" -> (outputFile <> ".gates.wl"),
  "nam_gate_count" -> Null,
  "nam_status" -> "not lowered; native counts include markers and are not NAM counts"
|>;
Export[outputFile, result, "RawJSON"];
Print[result];
Exit[If[status === "verified", 0, 5]];
