(* ::Package:: *)

(* Mathematica Package *)

BeginPackage["SymBuild`"]
(* Exported symbols added here with SymbolName::usage *)  


sparseArrayGlueRight::usage="sparseArrayGlueRight[matrix1, matrix2] glues the two sparse matrices with the same number of rows, placing them left and right.";
sparseArrayGlue::usage="sparseArrayGlue[matrix1, matrix2] glues two matrices with the same number of columns, placing them top and bottom. If the function is instead given an array of matrices with the same number of columns, then it glues them all. ";
sparseArrayZeroRowCut::usage="sparseArrayZeroRowCut[matrix] removes the zero rows at the bottom of the sparse matrix.";




(* ::Input::Initialization:: *)
Begin["`Private`"] (* Begin Private Context *)

End[] (* End Private Context *)


(*Sparse Matrix Manipulation*)

(*---------------------------------------------------------------------*)


sparseArrayGlueRight::nnarg=" The dimensions of the matrices are mismatched ";

sparseArrayGlueRight[A1_,A2_]/;If[Dimensions[A1][[1]]== Dimensions[A2][[1]],True,Message[sparseArrayGlueRight::nnarg];False]:=SparseArray[Union[A1//ArrayRules,(A2//ArrayRules)/.{a1_,a2_}:> {a1,a2+Dimensions[A1][[2]]}/;!(a1===_)],{Dimensions[A1][[1]],Dimensions[A1][[2]]+Dimensions[A2][[2]]}];

(*---------------------------------------------------------------------*)

sparseArrayGlue::nnarg=" The dimensions of the matrices are mismatched ";

sparseArrayGlue[A1_,A2_]/;If[Dimensions[A1][[2]]== Dimensions[A2][[2]],True,Message[sparseArrayGlue::nnarg];False]:=SparseArray[Union[A1//ArrayRules,(A2//ArrayRules)/.{a1_,a2_}:> {a1+Dimensions[A1][[1]],a2}/;!(a1===_)],{Dimensions[A1][[1]]+Dimensions[A2][[1]],Dimensions[A1][[2]]}];

sparseArrayGlue[A_]:=Which[Length[A]==0,A,Length[A]==1,A[[1]], Length[A]==2,sparseArrayGlue[A[[1]],A[[2]]],Length[A]>2, sparseArrayGlue[Join[{sparseArrayGlue[A[[1]],A[[2]]]},Drop[A,2]]]];


(*---------------------------------------------------------------------*)

sparseArrayZeroRowCut[sarray_]:=Module[{entriesPosition=Drop[(sarray//ArrayRules)[[All,1]],-1],dimArray=Dimensions[sarray]},If[entriesPosition==={},SparseArray[sarray,{1,dimArray[[2]]}],SparseArray[sarray,{Max[entriesPosition[[All,1]]],dimArray[[2]]}]]
];


EndPackage[]
