(* ::Package:: *)

(* ::Title:: *)
(*Loading the package (change the path of SymBuild.wl)*)


(* ::Section:: *)
(*Loading 'SymBuild' and *)


Get["/Users/Beryllium/SymbolBuilding-development/SymBuild.wl"];


(*---------------------------------------------------------------------------*)
(* Set to False if you don't want to use SpaSM, It is False by default*)
globalSpaSMSwitch=True;
globalSpaSMNumberOfKernels=8;

(* Set to true if you want SymBuild to parallelize some operations *)
globalSymBuildParallelize=True;
LaunchKernels[2];

(* Set to False to not have any messages appear (this variable is true by default, i.e. there are messages) *)
globalVerbose=True;


(* ::Section:: *)
(*Loading SpaSM if needed (load after the 'SymBuild' package )*)


(*
(*---------------------------------------------------------------------------*)
(* Loading the package *) *)

Import[$HomeDirectory<>"/Dropbox/packages/FF_lift/Eculidean_lift.wl"];
Import[$HomeDirectory<>"/Dropbox/packages/FF_lift/FF_linear_algebra_v2.wl"];

(* The paths for SpaSM: exchange and bench *)
SpaSMExchangePath = StringJoin[$HomeDirectory, "/exchange"]
SpaSMPath = StringJoin[$HomeDirectory, "/packages/spasm/bench"]








(* ::Section:: *)
(*Additional SymBuild global variables and parameters*)


(* You can play around with these configurations if you want *)
(* Using the command ??variable name, you should get an idea of what the parameter does *)

(*globalLowerThreshold=200; 
globalSpaSMThreshold=10000;
globalGetNullSpaceStep=200;
globalSpaSMListOfPrimes=Select[Range[2^14]+10000,PrimeQ];
globalGetNullSpaceSpaSMPrimes=Take[globalSpaSMListOfPrimes,-4];
globalSetOfBigPrimes=Select[2^63-Range[983],PrimeQ];
globalRowReduceOverPrimesInitialNumberOfIterations=2;
globalRowReduceOverPrimesMaxNumberOfIterations=10;
globalRowReduceOverPrimesMethod="Random"; 
globalRowReduceMatrixSpaSMPrimes=Take[globalSpaSMListOfPrimes,-4];*)

(* The command resetTheGlobalParameters[] resets everything to standard *)
globalGetNullSpaceSpaSMPrimes=Take[globalSpaSMListOfPrimes,-8];
globalRowReduceMatrixSpaSMPrimes=Take[globalSpaSMListOfPrimes,-8];


(* ::Title:: *)
(*Run this part to generate all up to weight 4 (30 min of computation )*)


(* ::Subchapter:: *)
(*Computing integrable symbols for the non-planar 5-pt alphabet (even+odd)*)


varNonPlanar={v[1],v[2],v[3],v[4],v[5]};
alphabetNonPlanar={v[1],v[2],v[3],v[4],v[5],v[3]+v[4],v[4]+v[5],v[1]+v[5],v[1]+v[2],v[2]+v[3],v[1]-v[4],v[2]-v[5],-v[1]+v[3],-v[2]+v[4],-v[3]+v[5],v[1]+v[2]-v[4],v[2]+v[3]-v[5],-v[1]+v[3]+v[4],-v[2]+v[4]+v[5],v[1]-v[3]+v[5],-v[1]-v[2]+v[3]+v[4],-v[2]-v[3]+v[4]+v[5],v[1]-v[3]-v[4]+v[5],v[1]+v[2]-v[4]-v[5],-v[1]+v[2]+v[3]-v[5],(-\[Epsilon]5+v[1] v[2]-v[2] v[3]+v[3] v[4]-v[1] v[5]-v[4] v[5])/(\[Epsilon]5+v[1] v[2]-v[2] v[3]+v[3] v[4]-v[1] v[5]-v[4] v[5]),(-\[Epsilon]5-v[1] v[2]+v[2] v[3]-v[3] v[4]-v[1] v[5]+v[4] v[5])/(\[Epsilon]5-v[1] v[2]+v[2] v[3]-v[3] v[4]-v[1] v[5]+v[4] v[5]),(-\[Epsilon]5-v[1] v[2]-v[2] v[3]+v[3] v[4]+v[1] v[5]-v[4] v[5])/(\[Epsilon]5-v[1] v[2]-v[2] v[3]+v[3] v[4]+v[1] v[5]-v[4] v[5]),(-\[Epsilon]5+v[1] v[2]-v[2] v[3]-v[3] v[4]-v[1] v[5]+v[4] v[5])/(\[Epsilon]5+v[1] v[2]-v[2] v[3]-v[3] v[4]-v[1] v[5]+v[4] v[5]),(-\[Epsilon]5-v[1] v[2]+v[2] v[3]-v[3] v[4]+v[1] v[5]-v[4] v[5])/(\[Epsilon]5-v[1] v[2]+v[2] v[3]-v[3] v[4]+v[1] v[5]-v[4] v[5]),\[Epsilon]5};
minimalPolynomialNonPlanar={\[Epsilon]5^2-\[CapitalDelta]};
replacementRuleNonPlanar={\[CapitalDelta]->v[1]^2 v[2]^2-2 v[1] v[2]^2 v[3]+v[2]^2 v[3]^2+2 v[1] v[2] v[3] v[4]-2 v[2] v[3]^2 v[4]+v[3]^2 v[4]^2-2 v[1]^2 v[2] v[5]+2 v[1] v[2] v[3] v[5]+2 v[1] v[2] v[4] v[5]+2 v[1] v[3] v[4] v[5]+2 v[2] v[3] v[4] v[5]-2 v[3] v[4]^2 v[5]+v[1]^2 v[5]^2-2 v[1] v[4] v[5]^2+v[4]^2 v[5]^2 };


timeMeasure=SessionTime[];
FtensorNonPlanar=computeTheIntegrabilityTensor[alphabetNonPlanar,varNonPlanar,{\[Epsilon]5},minimalPolynomialNonPlanar,replacementRuleNonPlanar,200,35,10]
Print["The computation took ", SessionTime[]-timeMeasure, " seconds."]
Clear[timeMeasure]


(* ::Subsubsection:: *)
(*Setting up the weight 1 symbols + first entry conditions + even/odd symbols*)


alphabetSigns=Join[Table[0,25],Table[1,5],Table[0,1]]
forbiddenFirstEntries=Complement[Range[31],Join[Range[5],15+Range[5]]]

{tensorLists[1],signs[1]}=weight1SolutionEvenAndOdd[alphabetNonPlanar,alphabetSigns,forbiddenFirstEntries];
presentIntegrableSymbolsData[{tensorLists[1],signs[1]}]


(* ::Subsubsection:: *)
(*Weight 2 with forbidden second entries*)


forbiddenSecondEntries={{1,8},{1,9},{1,14},{1,15},{1,24},{1,25},{2,9},{2,10},{2,11},{2,15},{2,21},{2,25},{3,6},{3,10},{3,11},{3,12},{3,21},{3,22},{4,6},{4,7},{4,12},{4,13},{4,22},{4,23},{5,7},{5,8},{5,13},{5,14},{5,23},{5,24},{16,8},{16,10},{16,11},{16,14},{16,21},{16,24},{17,6},{17,9},{17,12},{17,15},{17,22},{17,25},{18,7},{18,10},{18,11},{18,13},{18,21},{18,23},{19,6},{19,8},{19,12},{19,14},{19,22},{19,24},{20,7},{20,9},{20,13},{20,15},{20,23},{20,25}};

secondEntryEquations=weightLForbiddenSequencesEquationMatrix[{tensorLists[1]},forbiddenSecondEntries,31];
{tensorLists[2],signs[2]}=determineNextWeightSymbols[tensorLists[1],signs[1],FtensorNonPlanar,alphabetSigns,secondEntryEquations];
presentIntegrableSymbolsData[{tensorLists[2],signs[2]}]


(* ::Subsubsection:: *)
(*Weight 2 with no forbidden second entries*)


{tensorListsNoSE[2],signsNoSE[2]}=determineNextWeightSymbols[tensorLists[1],signs[1],FtensorNonPlanar,alphabetSigns];
presentIntegrableSymbolsData[{tensorListsNoSE[2],signsNoSE[2]}]


(* ::Subsubsection:: *)
(*Weight 3 with the forbidden second entries and no further conditions*)


{tensorLists[3],signs[3]}=determineNextWeightSymbols[tensorLists[2],signs[2],FtensorNonPlanar,alphabetSigns];
presentIntegrableSymbolsData[{tensorLists[3],signs[3]}]


(* ::Subsubsection:: *)
(*Weight 4 with the forbidden second entries and no further conditions*)


{tensorLists[4],signs[4]}=determineNextWeightSymbols[tensorLists[3],signs[3],FtensorNonPlanar,alphabetSigns];
presentIntegrableSymbolsData[{tensorLists[4],signs[4]}]


timeMeasure=AbsoluteTime[];
{tensorLists[5],signs[5]}=determineNextWeightSymbols[tensorLists[4],signs[4],FtensorNonPlanar,alphabetSigns];
Export["~/packages/SymbolBuilding-development/ISData.txt",presentIntegrableSymbolsData[{tensorLists[5],signs[5]}]//InputForm//ToString];
Print[AbsoluteTime[]-timeMeasure];

