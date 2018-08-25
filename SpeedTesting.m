(* ::Package:: *)

(* ::Title::Initialization:: *)
(*Loading the package*)


(* ::Section::Initialization:: *)
(*Loading 'SymBuild' and *)


(* ::Input::Initialization:: *)
testDirectory="/home/vladimir/Documents/Travail/BuildingSymbols/Mathematica/";


(* ::Input::Initialization:: *)
Get[testDirectory<>"SymBuild.wl"];


(* ::Input::Initialization:: *)
(*---------------------------------------------------------------------------*)
(* Set to False if you don't want to use SpaSM, It is False by default*)
(*globalSpaSMSwitch=True;
globalSpaSMNumberOfKernels=4;
*)
(* Set to true if you want SymBuild to parallelize some operations *)
globalSymBuildParallelize=True;
globalVerbose=False;
LaunchKernels[2];


(* ::Section::Initialization:: *)
(*Loading SpaSM if needed (load after the 'SymBuild' package )*)


(* ::Input::Initialization:: *)
(*
(*---------------------------------------------------------------------------*)
(* Loading the package *)
Get["/home/vladimir/SpaSM/src/spasm_m/spasm_m.wl"];
Import["/home/vladimir/SpaSM/src/Farey_fraction/Eculidean_lift.wl"];
Import["/home/vladimir/SpaSM/src/Modular_Linear_Algebra/FF_linear_algebra.wl"];

(* The paths for SpaSM: exchange and bench *)
SpaSMExchangePath="/home/vladimir/SpaSM/exchange";
SpaSMPath="/home/vladimir/SpaSM/spasm/bench";

(* This is needed to make the two packages work integrate with one another. *)
globalSpaSMExchangePath=SpaSMExchangePath;
*)







(* ::Section::Initialization:: *)
(*Additional SymBuild global variables*)


(* ::Input::Initialization:: *)
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


(* ::Title::Initialization:: *)
(*Functionality examples*)


(* ::Subchapter::Initialization:: *)
(*Computing integrable symbols for the planar 5-pt alphabet*)


(* ::Subsubsection::Initialization:: *)
(*Setup*)


(* ::Input::Initialization:: *)
varPlanar={v[1],v[2],v[3],v[4],v[5]};
alphabetPlanar={v[1],v[2],v[3],v[4],v[5],v[3]+v[4],v[4]+v[5],v[1]+v[5],v[1]+v[2],v[2]+v[3],v[1]-v[4],v[2]-v[5],-v[1]+v[3],-v[2]+v[4],-v[3]+v[5],v[1]+v[2]-v[4],v[2]+v[3]-v[5],-v[1]+v[3]+v[4],-v[2]+v[4]+v[5],v[1]-v[3]+v[5],(-\[Epsilon]5+v[1] v[2]-v[2] v[3]+v[3] v[4]-v[1] v[5]-v[4] v[5])/(\[Epsilon]5+v[1] v[2]-v[2] v[3]+v[3] v[4]-v[1] v[5]-v[4] v[5]),(-\[Epsilon]5-v[1] v[2]+v[2] v[3]-v[3] v[4]-v[1] v[5]+v[4] v[5])/(\[Epsilon]5-v[1] v[2]+v[2] v[3]-v[3] v[4]-v[1] v[5]+v[4] v[5]),(-\[Epsilon]5-v[1] v[2]-v[2] v[3]+v[3] v[4]+v[1] v[5]-v[4] v[5])/(\[Epsilon]5-v[1] v[2]-v[2] v[3]+v[3] v[4]+v[1] v[5]-v[4] v[5]),(-\[Epsilon]5+v[1] v[2]-v[2] v[3]-v[3] v[4]-v[1] v[5]+v[4] v[5])/(\[Epsilon]5+v[1] v[2]-v[2] v[3]-v[3] v[4]-v[1] v[5]+v[4] v[5]),(-\[Epsilon]5-v[1] v[2]+v[2] v[3]-v[3] v[4]+v[1] v[5]-v[4] v[5])/(\[Epsilon]5-v[1] v[2]+v[2] v[3]-v[3] v[4]+v[1] v[5]-v[4] v[5]),\[Epsilon]5};
minimalPolynomialPlanar={\[Epsilon]5^2-\[CapitalDelta]};
replacementRulePlanar={\[CapitalDelta]->v[1]^2 v[2]^2-2 v[1] v[2]^2 v[3]+v[2]^2 v[3]^2+2 v[1] v[2] v[3] v[4]-2 v[2] v[3]^2 v[4]+v[3]^2 v[4]^2-2 v[1]^2 v[2] v[5]+2 v[1] v[2] v[3] v[5]+2 v[1] v[2] v[4] v[5]+2 v[1] v[3] v[4] v[5]+2 v[2] v[3] v[4] v[5]-2 v[3] v[4]^2 v[5]+v[1]^2 v[5]^2-2 v[1] v[4] v[5]^2+v[4]^2 v[5]^2 };


(* ::Input::Initialization:: *)
FtensorPlanar=computeTheIntegrabilityTensor[alphabetPlanar,varPlanar,{\[Epsilon]5},minimalPolynomialPlanar,replacementRulePlanar,200,35,10];



(* ::Subsubsection::Initialization:: *)
(*Setting up the weight 1 symbols + first entry conditions + even/odd symbols*)


(* ::Input::Initialization:: *)
alphabetSigns=Join[Table[0,20],Table[1,5],Table[0,1]];
forbiddenFirstEntries=Complement[Range[26],Range[5]];

{tensorLists[1],signs[1]}=weight1SolutionEvenAndOdd[alphabetPlanar,alphabetSigns,forbiddenFirstEntries];


(* ::Subsubsection::Initialization:: *)
(*Weight 2*)


(* ::Input::Initialization:: *)
timeMeasure=SessionTime[];


(* ::Input::Initialization:: *)
{tensorLists[2],signs[2]}=determineNextWeightSymbols[tensorLists[1],signs[1],FtensorPlanar,alphabetSigns];


(* ::Input::Initialization:: *)
timeMeasure=SessionTime[]-timeMeasure;
Put[{timeMeasure,tensorLists[2],signs[2]},testDirectory<>"testWeight2.txt"]


(* ::Subsubsection::Initialization:: *)
(*Weight 3*)


(* ::Input::Initialization:: *)
timeMeasure=SessionTime[];


(* ::Input::Initialization:: *)
{tensorLists[3],signs[3]}=determineNextWeightSymbols[tensorLists[2],signs[2],FtensorPlanar,alphabetSigns];


(* ::Input::Initialization:: *)
timeMeasure=SessionTime[]-timeMeasure;
Put[{timeMeasure,tensorLists[3],signs[3]},testDirectory<>"testWeight3.txt"]


(* ::Subsubsection::Initialization:: *)
(*Weight 4*)


(* ::Input::Initialization:: *)
timeMeasure=SessionTime[];


(* ::Input::Initialization:: *)
weightHere=4;


(* ::Input::Initialization:: *)
{tensorLists[weightHere],signs[weightHere]}=determineNextWeightSymbols[tensorLists[weightHere-1],signs[weightHere-1],FtensorPlanar,alphabetSigns];


(* ::Input::Initialization:: *)
timeMeasure=SessionTime[]-timeMeasure;
Put[{timeMeasure,tensorLists[weightHere],signs[weightHere]},testDirectory<>"testWeight"<>ToString[weightHere]<>".txt"]


(* ::Subsubsection::Initialization:: *)
(*Weight 5*)


(* ::Input::Initialization:: *)
timeMeasure=SessionTime[];


(* ::Input::Initialization:: *)
weightHere=5;


(* ::Input::Initialization:: *)
{tensorLists[weightHere],signs[weightHere]}=determineNextWeightSymbols[tensorLists[weightHere-1],signs[weightHere-1],FtensorPlanar,alphabetSigns];


(* ::Input::Initialization:: *)
timeMeasure=SessionTime[]-timeMeasure;
Put[{timeMeasure,tensorLists[weightHere],signs[weightHere]},testDirectory<>"testWeight"<>ToString[weightHere]<>".txt"]


(* ::Subsubsection::Initialization:: *)
(*Weight 6*)


(* ::Input::Initialization:: *)
timeMeasure=SessionTime[];


(* ::Input::Initialization:: *)
weightHere=6;


(* ::Input::Initialization:: *)
{tensorLists[weightHere],signs[weightHere]}=determineNextWeightSymbols[tensorLists[weightHere-1],signs[weightHere-1],FtensorPlanar,alphabetSigns];


(* ::Input::Initialization:: *)
timeMeasure=SessionTime[]-timeMeasure;
Put[{timeMeasure,tensorLists[weightHere],signs[weightHere]},testDirectory<>"testWeight"<>ToString[weightHere]<>".txt"]


(* ::Subsubsection:: *)
(*End*)


(* ::Input::Initialization:: *)
Quit[]
