(* ::Package:: *)

(* ::Title::Initialization:: *)
(*(*Beginning/ First declarations *)*)


(*----------------------------------------------------------------------------------------------------------------------------------*)
(*----------------------------------------------------------------------------------------------------------------------------------*)
(* Mathematica Package: SymBuild *)
(*
    Copyright (C) Vladimir Mitev and Yang Zhang.
    The program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 2 as
    published by the Free Software Foundation.

    The program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
*)

(*----------------------------------------------------------------------------------------------------------------------------------*)
(*----------------------------------------------------------------------------------------------------------------------------------*)

BeginPackage["SymBuild`"]

Print["SymBuild: Mathematica package for the contruction and manipulation of integrable symbols in scattering amplitudes. "]
Print["Version 0.37, July 24, 2018 "]
Print["Created by: Vladimir Mitev and Yang Zhang, Johannes Gutenberg University of Mainz, Germany. "]



(* ::Title::Initialization:: *)
(*(*Descriptions of the exported symbols*)*)


(*----------------------------------------------------------------------------------------------------------*)
(* The descriptions of the exported symbols are added here with 'SymbolName::usage' *)
(* They are divided into subsection according to their use *)  
(*----------------------------------------------------------------------------------------------------------*)



(* ::Chapter:: *)
(*Protected symbols*)


Protect[derivative,\[ScriptCapitalS]];


(* ::Chapter::Initialization:: *)
(*(*General command on lists and matrices manipulations *)*)


(*----------------------------------------------------------------------------------------------------------*)
(*Sparse Matrix Manipulation*)
sparseArrayGlueRight::usage="sparseArrayGlueRight[matrix1, matrix2] glues the two sparse matrices with the same number of rows, placing them left and right.";
sparseArrayGlue::usage="sparseArrayGlue[matrix1, matrix2] glues two matrices with the same number of columns, placing them top and bottom. If the function is instead given an array of matrices with the same number of columns, then it glues them all. ";
sparseArrayZeroRowCut::usage="sparseArrayZeroRowCut[matrix] removes the zero rows at the bottom of the sparse matrix.";
transposeLevelsSparseArray::usage="transposeLevelsSparseArray[matrix, level specification] generalizes matrix transposition to a sparse array with many indices. It can be used in two ways. Either as 'transposeLevelsSparseArray[sparseArray_,level1_,level2_]' in which case it takes a tensor with index structure T[i_1,....i_level1,...i_level2...] and turns it into T[i_1,....i_level2,...i_level1...] or as  'transposeLevelsSparseArray[sparseArray_,swappedOrder_]' in which case swappedOrder={neworder_1, ..., neworder_sizearray} gives the new position of the indices. ";

dotSymbolTensors::usage="dotSymbolTensors contracts the last entry of the R-tensor tensorL with the first entry of the Q-tensor tensorR to produce a new (R+Q-2) tensor. 
It can be used as 'dotSymbolTensors[tensorL_,tensorR_]' in which case it contracts the two arrays or as 'dotSymbolTensors[A_]',
 where A={tensor1, tensor2,...} in which case it gives tensor1.tensor2.tensor3.... ";

(*----------------------------------------------------------------------------------------------------------*)
(*dense Matrix Manipulation*)
denseMatrixConcatenateBelow::usage="denseMatrixConcatenateBelow[matrix1, matrix2] glues the two dense matrices with the same number of columns, placing them top and bottom. This command is used in particular in 'collectRootCoefficients'. ";
denseMatrixConcatenateRight::usage="denseMatrixConcatenateRight[matrix1, matrix2] glues the two dense matrices with the same number of rows, placing them left and right.";
denseMatrixZeroRowCut::usage="denseMatrixZeroRowCut[matrix] removes the zero rows at the bottom of the dense matrix.";

(*---------------------------------------------------------------------*)
positionDuplicates::usage="positionDuplicates[list] finds the positions of duplicates in a list and groups them.";



linearIndependentColumns::usage="linearIndependentColumns[matrix] gives a list containing the position of a linearly independent set of columns of the matrix.
 The matrix should not be too large, since this command uses the standard RowReduce command. ";


determineLeftInverse::usage="determineLeftInverse[sparseMatrix_] is a function that determines the left inverse of a sparse matrix that has more rows than columns.  ";


(* ::Chapter::Initialization:: *)
(*(*Symbol tensors and their manipulations*)*)


(*---------------------------------------------------------------------*)
integrableSymbolsTensorsGlue::usage="integrableSymbolsTensorsGlue[tensor1, tensor 2 ] takes two tensors describing integrable symbols and glues them into a common one. It is used in particular to glue the tables of even tensors and of odd tensors.";

(*---------------------------------------------------------------------*)

solutionSpaceToSymbolsTensor::usage="solutionSpaceToSymbolsTensor[sparseSolutionSpace_,sizeAlphabet_] transforms the solution (a matrix) of the integrability conditions into the set of tensors. Thus functions returns a 3 tensor with the index structure [jprevious=1,...,dimSolutionSpacePrevious, i=1,...,sizeAlphabet,j=1,...dimSolutionSpace].  ";


(*---------------------------------------------------------------------*)
symbolsTensorToVector::usage="This command takes a tensor (like d[All,All,somenumber]) and turns it into a vector that we feed into the left inverse matrix. This is used to expand a general integrable symbol in a basis.";


symbolsTensorToSolutionSpace::usage="The command 'symbolsTensorToSolutionSpace[symbolsTensor_]'  is the inverse function to 'solutionSpaceToSymbolsTensor' and it takes the symbol tensors and gives back the matrix whose rows are in the kernel of the integrability conditions. ";

expressTensorAsSymbols::usage=" The command 'expressTensorAsSymbols[sparseArrayRules_,options_.]' expresses lists of integrable tensors explicitly as formal symbols. 
The variable 'sparseArrayRules' is a tensor obtained by using the command 'dotSymbolTensors' on an array of integrable symbols {d_{M},....d_L} starting at weight M and continuing to weight L. 
Then the command takes this tensor with L-M+3 indices and expresses it a list of formal symbols. 

The expression can be made in 3 different ways: If 'options' is the string 'Recursive'' then the output is of the type 
'Sum of  sym[S[j_{M-1}],{i_{M},i_{M+1},...,i_L}]' where S[j_{M-1}] are the integrable symbols of weight M and the remainig i_s indices indicate the attached letters.
If 'options= 'Complete'', then the S[j_{M-1}] is replaced by an index in the remaining array, like this : 'Sum of sym[{i_{M-1},i_{M},...,i_L}]'. This is only meaningful if M=2 or M=1. 
If 'options= 'DropFirst'', then the output is similar to the 'Complete' case bu the first entry of the array is ignored and the output is of the type 'Sum of sym[{i_{M},i_{M+1},...}]'. 
This is only ok for M=1";



(* ::Chapter::Initialization:: *)
(*(*Formal symbols manipulation*)*)


(* ::Input::Initialization:: *)
(*---------------------------------------------------------------------*)
shuffles::usage="shuffles[list1, list2] gives a list that contains all the shuffles of the lists 1 and 2. ";

(*---------------------------------------------------------------------*)
shuffleProduct::usage="shuffleProduct[symbol1, symbol2] takes two formal sums of symbols SB[___], multiplies them together and replaces the products of SB[___] by the appropriate shuffle products. ";

SB::usage="SB[list] is a formal way of writing integrable symbols. It satisfies the properties: 1) SB[{..., const,...}]=0 2) SB[{..., A B ,...}]=SB[{...,A,...}]+SB[{...,B,...}] and 3) SB[{...,A^(-1),...}]=-SB[{...,A,...}]. ";


(*---------------------------------------------------------------------*)
listifySum::usage="listifySum[expression] takes a sum A_1+....+A_n and turns it into a list {A1,...,A_n}. If applied to an expression A that is not a sum it just returns {A}.";


(*---------------------------------------------------------------------*)
factorSymbols::usage="factorSymbols[expression] takes a formal sum of SB[A] and then factors the list A. Furthermore, it applies factor lists so as to for example identify entries in the lists like '1-x' and 'x-1'. ";


(*---------------------------------------------------------------------*)
extractSingularPart::usage="extractSingularPart[sum of symbols, variable] takes a formal sum of symbols SB[A] and extracts those that are logarithmically singular in the limit variable -> 0. ";

(*---------------------------------------------------------------------*)
takeLimitOfSymbol::usage="takeLimitOfSymbol[sum of symbols, variable] takes a formal sum of symbols SB[A] and expands in the limit variable->0. In such a limit (x->0) SB[x+y,...]  becomes SB[y,...] but SB[x,...] remains SB[x,...] because the latter is a logarithmic singularity. ";

(*---------------------------------------------------------------------*)

expandInSymbolBasis::usage="The function expandInSymbolBasis[exp_,basis_] takes an expression exp=Sum_i c_i SB[list_i] and expands it in the basis 'basis={Sum_i a_i SB[list_i],....}. 
The answer is the list of coefficients in the expansion. ";

(*---------------------------------------------------------------------*)
subProductProjection::usage="subProductProjection[symbol]  takes a symbol SB[A] and sends to zero the part of SB[A] that is can be written as a product of symbols of lower weight. 
See for example arXiv:1401.6446 for the definition of the map. ";

(*---------------------------------------------------------------------*)
productProjection::usage="productProjection[symbol] applies 'subProductProjection' to each element of a sum of explicit formal symbols SB[list_]. 
This projects away all the elements that can be written as products of symbols of lower weight. ";

convertFormalSymbol::usage="The function 'convertFormalSymbol[expression_,alphabet_]' converts a sum of formal symbols sb[list_] in a given alphabet into a sum of explicit symbols SB[list_] with the entries of the list replaced by the explicit letters of the alphabet. ";


(*---------------------------------------------------------------------*)
IntegrableQ::usage="The function 'IntegrableQ[expression_,FtensorOrVariables_]' checks if an expression of formal symbols sb[] or SB[] is integrable. It returns True if that is the case.
 When acting on expressions of SB formal symbols, the second entry of IntegrableQ must be a list of variables. 
When acting on expressions of sb formal symbols, the second entry must be an integrability tensor. "; 




(* ::Chapter:: *)
(*Commands used in checking the independence of the alphabet*)


(*---------------------------------------------------------------------*)

dLogAlphabet::usage="Compute the dLog of an alphabet (with roots). Used then in the command 'findRelationsInAlphabet' to determine if an alphabet is independent or not. ";

findRelationsInAlphabet::usage="The command 'findRelationsInAlphabet[alphabet_,allvariables_,listOfRoots_,listOfRootPowers_,sampleSize_,maxSamplePoints_,toleranceForRetries_]' 
determines if the dlog of the functions in 'alphabet' are linearly independent or not. 
The parameters 'sampleSize_,maxSamplePoints_,toleranceForRetries_' play the same role as in the command 'buildFMatrixReducedIterativelyForASetOfEquations'. 
If the alphabet is not idenpendent, the command will generate a matrix whose rows are the linear combinations of letters that are zero. ";



(* ::Chapter:: *)
(*Computing the difference equation if given a sequence of dimensions*)


(*---------------------------------------------------------------------*)
computeCoefficientsOfDifferenceEquation::usage="The function computeAlphas[dimSequence_] takes a sequence {dimH[0],dimH[1], dimH[2],...} of the dimensions of all integrable symbols 
at given weight (up to some cutoff) and attemps to guess a sequence of numbers {\[Alpha]_0=1,\[Alpha]_1, \[Alpha]_2,...  \[Alpha]_s} such that Sum[\[Alpha]_r (-1)^{r} dimH[L-r] ,{r,0,s}]=0.
 This provides a difference equation that the dimensions of the spaces of integrable symbols have to satisfy.  ";



(* ::Chapter:: *)
(*Counting the number of products and of irreducible symbols*)


(*---------------------------------------------------------------------*)
rewritePartition::usage="The function 'rewritePartition[partition_]' takes an integer partition of N, i.e. a list 'partition'= {n_1,n_2,n_3,...n_r} with N=Sum_i n_i and n_i>= n_{i+1}, and rewrites it as a table {m_1,m_2,...,m_s}, where m_j is the number of times the number j appears in 'partition' and s is the largest number that appears in 'partition'. For instance, rewritePartition[{5,1,1}]={2,0,2,0,1}. ";

dimProductSymbols::usage="The function dimProductSymbols[L_] gives the number of integrable symbols at weight L that are products. The answer is given as a function of 'dimQ[n]' which is the number of 'irreducible' symbols of weight n. ";

dimIrreducibleSymbols::usage="The function dimIrreducibleSymbols[cutoffWeight_] gives a table {dimQ[0], dimQ[1], ..., dimQ[cutoffWeight]} of the number of irreducible integrable symbols (i.e. those that cannot be written as products) of weight smaller or equal to cutoffWeight. The answer is given as a function of dimH[n], which is the total number of integrable symbols of weight n.";


(* ::Chapter:: *)
(*Computing the integrability tensor \[DoubleStruckCapitalF]*)


(*---------------------------------------------------------------------*)
matrixFReducedToTensor::usage="matrixFReducedToTensor[sparse matrix] transforms a R x Binomial[len,2] sparse matrix into a R x len x len tensor. This is used to transform the \[DoubleStruckCapitalM] matrix into the \[DoubleStruckCapitalF] tensor that then enters into the computations of the integrable symbols.  Here, 'len' is the length of the alphabet which the command inferres from the size of the matrix. ";

integrableEquationsRational::usage="integrableEquationsRational[alphabet, list of variables] takes an alphabet of rational functions in the variables and generates a matrix of size Binomial[number of variables, 2] x Binomial[ length of the alphabet, 2], each entry of which is a rational function.  ";

integrableEquationsWithRoots::usage="integrableEquationsWithRoots[alphabet, list of variables, list of roots, list of root powers] takes an alphabet of rational functions in the variables and in the roots and generates a matrix of size Binomial[number of variables, 2] x Binomial[ length of the alphabet, 2]. The 'list of roots', should be a list of rational functions \!\(\*SubscriptBox[\(\[CapitalDelta]\), \(i\)]\) in the variables and the 'list of root powers' contains the powers \!\(\*SubscriptBox[\(n\), \(i\)]\) such that root[i\!\(\*SuperscriptBox[\(]\), SubscriptBox[\(n\), \(i\)]]\) = \!\(\*SubscriptBox[\(\[CapitalDelta]\), \(i\)]\). The roots should be represented in the alphabet as formal functions root[i][variable 1, variable 2,....]. ";


(*---------------------------------------------------------------------*)
(*Resolve the roots using Gr\[ODoubleDot]bner bases*)
resolveRootViaGroebnerBasis::usage="resolveRootViaGroebnerBasis[expressionToSimplify, list of roots, list of root powers] takes a rational function 
in the formal root[i][variable1, variable2,...] (such as the entries of the matrix generated by 'integrableEquationsWithRoots') and uses Gr\[ODoubleDot]bner bases to simplify them, 
such as the output is of the form: Sum[r[s1,s2,...] \[Rho]1^s1 \[Rho]2^s2...,{s1,0,n1-1},{s2,0,n2-1}], where the powers n1, n2,... satisfy the conditions
root[i]^(ni) = \[CapitalDelta]i. Here, 'list of roots' = {\[CapitalDelta]1,...} and 'list of root powers' ={n1,n2,...}. ";

(*---------------------------------------------------------------------*)
resolveRootViaGroebnerBasisMatrix::usage="resolveRootViaGroebnerBasisMatrix[matrix , list of roots, list of root powers] applies 'resolveRootViaGroebnerBasis' to each element of an array. For our purposes, the array will be a dense one. ";


(*---------------------------------------------------------------------*)
(*Generating the integrability matrix \[DoubleStruckCapitalM] from a list of rational equations*)
buildFMatrixReducedForASetOfEquations::usage="buildFMatrixReducedForASetOfEquations[setOfEquations_,allvariables_,sampleSize_,maxSamplePoints_,toleranceForRetries_] 
takes first a matrix of rational functions in the 'list of variables'. Proceeding row by row, the command samples the functions over random prime numbers of a maximal
 size given by 'sampleSize'. If a division by zero happens randomly, it tries again a number of times given by 'tolerance for retries'.
 The number of sample points per row is given by 'maxSamplePoints'. After the sampling is done, the command 'rowReduceMatrix' to row reduce the sampled matrix and then the zero rows are removed. ";
 
 
 (*---------------------------------------------------------------------*)
(*Commands to use when the \[DoubleStruckCapitalM] matrix contains roots *) 
takeSecondEntry::usage="'takeSecondEntry[array_]' is an auxiliary command used for example in 'collectRootCoefficients'. 
It is applied to a list. If that list is of the type '{entry \[Rule] X}' it gives X and if the array is {} it gives zero. ";

(*---------------------------------------------------------------------*)
collectRootCoefficients::usage="The command collectRootCoefficients[expressionArray_,namesOfRoots_] start with an array 
expressionArray= { coefficients_{n1n2...} * \[Rho]1^n1 \[Rho]2^n2....., .....} and turns it into an array 
{coefficients_{n1n2...},..} such that each row corresponds to the same powers of the roots \[Rho]i. Here, 'namesOfRoots'={\[Rho]1,\[Rho]2,....} is the list of the different roots. ";

(*---------------------------------------------------------------------*)
(* The simplest end-user command for the computation of the integrability tensor *)
computeTheIntegrabilityTensor::usage="The command 'computeTheIntegrabilityTensor[alphabet_,allvariables_,listOfRoots_,listOfRootPowers_,sampleSize_,maxSamplePoints_,toleranceForRetries_]' computes the integrability tensor \[DoubleStruckCapitalF] for a given alphabet 'alphabet_' in the variables 'allvariables_'. The array 'listOfRoots_' contains a list of rational functions {\[CapitalDelta]1, ..., \[CapitalDelta]s} such that the roots \[Rho]i obey \[Rho]i^{ni}=\[CapitalDelta]i, where the powers ni are contained in the array 'listOfRootPowers_'. See the command 'buildFMatrixReducedForASetOfEquations' for an explanation of the parameters 'sampleSize_,maxSamplePoints_,toleranceForRetries_'.  ";




(* ::Chapter:: *)
(*Null space commands*)


(*---------------------------------------------------------------------*)
modifiedRowReduce::usage="modifiedRowReduce[sparse matrix] transforms a sparse array into a dense one and then applies row reduction on it. 
This is needed since acting with RowReduce on zero matrices can make the kernel crash. This is used for example in the command 'getNullSpaceStepByStep'. ";

(*---------------------------------------------------------------------*)
getNullSpaceFromRowReducedMatrix::usage="getNullSpaceFromRowReducedMatrix[row reduced sparse matrix] takes a sparse matrix A that has been brought into row echelon form and generates a matrix whose rows are a basis of the kernel of A. ";

getNullSpaceStepByStep::usage="getNullSpaceStepByStep[matrix, step] computes the null space of a matrix by dividing it into subpieces with 'step' number of rows. At each iteration the nullspace computed previously is plugged into the next subpiece which reduces the number of columns in the computation. ";


getNullSpace::usage=" The command 'getNullSpace[matrix_]' computes the null space of 'matrix'. 
If the number of rows of 'matrix' is smaller than the global variable 'globalLowerThreshold', it uses the standard Mathematica command NullSpace. 
If bigger than that number but smaller than 'globalSpaSMThreshold', it uses the command 'getNullSpace' which computes the null space iteratively after dividing the matrix into several small ones that have at most 'globalGetNullSpaceStep' rows. 
If the number of rows of 'matrix' is larger than the global variable 'globalSpaSMThreshold', then this command calls the external program SpaSM. The rows of the returned matrix are a basis for the null space. ";



(* ::Chapter:: *)
(*Row reduction of the finite fields*)


(*---------------------------------------------------------------------*)
(*Rational reconstruction*)
rationalReconstructionAlgorithm::usage="rationalReconstructionAlgorithm[q, prime number p] a reasonable guess for the fraction r in the field Q of the rationals such that r = q in the field of the prime number p. ";

rationalReconstructionArray::usage="rationalReconstructionArray[array, prime number p] applies the function rationalReconstructionAlgorithm[#,p] on each non-zero element of the array. ";

applyChineseRemainder::usage="The command 'applyChineseRemainder[matrixList_,primesList_]' takes the list of matrices 'matrixList = {M_1, M_2,... , M_Q}' and applies the chinese remainder algorithm using the primes in 'primesList={p_1,p_2,....p_Q}'.";

rowReduceOverPrimes::usage="Use the finite field reduction over primes. Start with 'globalRowReduceOverPrimesInitialNumberOfIterations' (mostly equal to 2) number of primes, then compare the reconstructions. If there is no majority opinion, keep adding primes and then constructing bigger rational reconstructions by using the chinese remainder algorithm. At each step, check if a majority of the reconstructions agree. If they do, pick the majority opinion. ";



(*---------------------------------------------------------------------*)
(*Row reduction (over the finite fields)*)
rowReduceMatrix::usage=" The command 'rowReduceMatrix[matrix_]' computes the row-eshelon form of a matrix. FINISH THE DESCRIPTION!! ";



(* ::Chapter:: *)
(*Determine the tranformation matrices between sets of integrable symbols*)


buildTransformationMatrix::usage="The command 'buildTransformationMatrix[weightLsymbolTensor_,previousTransformationMatrix_,alphabetTransformationMatrix_,limitAlphabetInversionMatrix_]' produces the weight L matrix T_L that gives the limit of the integrable symbols in the alphabet A1 in the alphabet A2. It takes  as input: 1) the matrix 'alphabetTransformationMatrix' which tells us how the limit of the letters of A1 are expressed in letters of the limit alphabet A2, 2) the symbol tensor 'weightLsymbolTensor' of the weight L integrable symbols in A1, 3) the matrix 'previousTransformationMatrix' (this is T_{L-1}) and 4) the matrix 'limitAlphabetInversionMatrix' which is the weight L inversion matrix for the integrable symbols in the alphabet A2.  ";


inverseMatrixToTensor::usage="The command 'inverseMatrixToTensor[inverseMatrix_,sizeAlphabet_]' takes the weight L inversion matrix E_L for the integrable symbols in a certain alphabet A and transforms it into a 3-tensor by splitting the column index in a bi-index (j,k) where k=1,... Length[A]. ";


(* Two auxiliary commands *)
auxFlattenTwoIndices12::usage="The auxiliary command 'auxFlattenTwoIndices12[sparsearray_,sizeAlphabet_]' flattens the first two indices of a 3-tensor with indices (i,j,k), where the second index j is in the interval j=1,..., sizeAlphabet. ";

auxFlattenTwoIndices23::usage="The auxiliary command 'auxFlattenTwoIndices23[sparsearray_,sizeAlphabet_]' flattens the last two indices of a 3-tensor with indices (i,j,k), where the third index k is in the interval k=1,..., sizeAlphabet. ";



(* ::Chapter:: *)
(*Computing the next level symbols*)


(*---------------------------------------------------------------------*)

nextWeightSymbolsEquationMatrix::usage="The function 'nextWeightSymbolsEquationMatrix[previousWeightSymbolsTensor_,FmatrixTensor_]' creates a matrix 
for the equations at the next level (level L) given the L-1 solution tensor 'previousWeightSymbolsTensor'  and the integrability tensor 'FmatrixTensor'. 
It is used in the end-user command 'determineNextWeightSymbols'. ";

(*---------------------------------------------------------------------*)

determineNextWeightSymbolsSimple::usage="The command 'determineNextWeightSymbolsSimple[previousWeightSymbolsTensor_,FmatrixTensor_]' computes the weight L solution (in tensor form) 
if given the weight L-1 solution and the integrability tensor \[DoubleStruckCapitalF]. This is a simple command - it does not care for even vs odd symbols nor for any additional forbidden entries beyond
 those put by hand at weight 1 by using the command 'weight1Solution'. ";
 
(*---------------------------------------------------------------------*)
(* The combination command for the even+odd symbol computation *)

 
determineNextWeightSymbols::usage="The command 'determineNextWeightSymbols[previousWeightSymbolsTensor_,previousWeightSymbolsSigns_,FmatrixTensor_,listOfSymbolSigns_,forbiddenSequenceConditions_.]' is 
the end-user command for the computation of the next weight (weight = L) integrable symbols (with optional forbidden entries) and their decomposition into even + odd symbols. The variable 'previousWeightSymbolsTensor'
is the weight L-1 integrable symbol tensor, the array 'previousWeightSymbolsSigns' is the list of 0 and 1 determining the signs of the symbols, 'FmatrixTensor' is the integrability tensor for the 
alphabet under consideration and 'listOfSymbolSigns' is the array of 0 and 1 determining the signs of the letters of the alphabet.  Finally, the optional sparse array 'forbiddenSequenceConditions' is a 
matrix of conditions determined using the command 'weightLForbiddenSequencesEquationMatrix' that forbid certain sequences of letters. By default, it is empty, meaning that no weight L entry condition
is imposed.  ";


(*---------------------------------------------------------------------*)
(* construction of the weight 1 integrable tensors*)
weight1Solution::usage="The command 'weight1Solution[alphabet_,forbiddenEntries_.]' creates the weight 1 solution in tensor form when given the alphabet and (optionally) a set of forbidden entries. By default the latter is an empty list.  ";

weight1SolutionEvenAndOdd::usage="The command 'weight1SolutionEvenAndOdd[alphabet_,listOfSymbolSigns_,forbiddenEntries_]' creates the weight 1 solution in tensor form when given the alphabet, 
a list that tells which letters are even/odd and (optionally) a set of forbidden entries (By default this is an empty list). 
The result is an array {weight1tensors, listOfSigns} for weight 1, where 'listOfSigns' is a 1D array of 0 or 1 with 0 meaning the corresponding symbol is even and 1 that it is odd. ";

(*---------------------------------------------------------------------*)
(* The n-entry conditions *)

weightLForbiddenSequencesEquationMatrix::usage=" The command 'weightLForbiddenSequencesEquationMatrix[allPreviousWeightSymbolsTensorList_,listOfForbiddenSequences_,sizeAlphabet_]' computes the 
matrix of conditions for the weight L integrable symbols such that certain sequences of letters (given by the variable 'listOfForbiddenSequences') do not appear. The array
'allPreviousWeightSymbolsTensorList' is a list of all the tensors of integrable symbols of lower weight, i.e. {d_1,d_2,d_3,....d_{L-1}}, where d_i are the weight i integrable symbol tensors. "

(*---------------------------------------------------------------------*)
(* Commands for the computations of the even+odd symbols*)
makeSparseMatrixOutOfIndexLists::usage="makeSparseMatrixOutOfIndexLists[index1_,index2_,size1_,size2_] makes a sparse matrix of size (Length[index1]*Length[index2])x (size1*size1) with entries 1 and 0. \[IndentingNewLine]The position of the 1s is given by the entries of index1 and index2 in a tensorial way. ";


makeTheEvenOddConditionsMatrix::usage="makeTheEvenOddConditionsMatrix[previousIntegrableSymbolsSigns_,listOfSymbolSigns_,evenOrOdd_] generates a matrix of combitions 
that have to be satisfied by the weight L even (for 'evenOrOdd'=0) or odd (for 'evenOrOdd'=1) integrable symbols, where 'previousIntegrableSymbolsSigns' 
is the tensor for the  weight L-1 integrable symbols and 'listOfSymbolSigns' is a list of zeroes or ones depending on the parity of the elements in 'previousIntegrableSymbolsSigns'. ";




(* ::Chapter:: *)
(*Taking derivatives of the symbols*)


symbolDerivative::usage="The operator 'symbolDerivative' can be used in two ways. First, it can be called with 2 entries as 'symbolDerivative[expression, variable]'
in which case it acts on an expression involving explicit symbols SB[A_] and takes their derivative w.r.t. 'variable'. In the second case, it is called
with 3 entries as 'symbolDerivative[expression, alphabet, variable]' in which case it acts on a sum of formal symbols sb[list_] or sb[\[ScriptCapitalS][_],list_] and
computes their derivative w.r.t. 'variable'. ";



(* ::Chapter:: *)
(*Presentation commands*)


presentIntegrableSymbolsData::usage="The command 'presentIntegrableSymbolsData[{tensorList_,signsArray_}]' takes an array of two elements, one being the tensor of integrable symbols and the other
the list of 0 and 1 indicating which symbols are even/odd, and presents the data in a nice way. Alternatively, it can be called as 'presentIntegrableSymbolsData[tensorList_]' 
in which case it does not care about the even/odd symbols. ";



(* ::Title::Initialization:: *)
(*(*Global variables: definitions and descriptions *)*)


(* ::Section:: *)
(*Parallelize*)


globalSymBuildParallelize::usage=" The variable 'globalSymBuildParallelize' determines whether parallelization takes place in SymBuild or not. By default, it is false. ";
globalSymBuildParallelize=False;


(* ::Section::Initialization:: *)
(*(*SpaSM global variables*)*)


(* ::Input::Initialization:: *)
(*"MatrixDirectory" is used in SpaSM!! DON'T OVERWRITE!*)
(*"Nkernel" is a parameter used in SpaSM! DON'T OVERWRITE! *)


(* ::Input::Initialization:: *)
globalSpaSMSwitch::usage=" This is a global parameter that specifies whether 'SpaSM' is used or not. By default it is true. Do not forget to set 'globalSpaSMExchangePath=SpaSMExchangePath'
in the notebook! ";
globalSpaSMSwitch=False;


globalSpaSMExchangePath::usage=" This is a global parameter that specifies the folder in which the temporary files used by SpaSM are to be stored. (YANG?)";globalSpaSMExchangePath="/home/vladimir/SpaSM/exchange";


globalSpaSMListOfPrimes::usage=" This is a global parameter that provides a list of primes that can be used in SpaSM. The primes used in that program should not be larger than \!\(\*SuperscriptBox[\(2\), \(16\)]\).";
globalSpaSMListOfPrimes=Select[Range[2^14]+10000,PrimeQ];

globalSpaSMNumberOfKernels::usage=" This is a global parameter that specifies the number of computer kernels that SpaSM will use. ";
globalSpaSMNumberOfKernels=2;



(* ::Section::Initialization:: *)
(*(*Global parameters for 'getNullSpace'*)*)


(* ::Input::Initialization:: *)
globalLowerThreshold::usage=" This is a global parameter in the command 'getNullSpace'. If the matrix whose null space must be computed has less rows that this number, then the standard 'NullSpace' command is used. ";
globalLowerThreshold=300; 

globalSpaSMThreshold::usage=" This is a global parameter in the command 'getNullSpace' (and 'rowReducedMatrix'). If the matrix whose NullSpace must be computed has less rows that this number but higher or equal than 'globalLowerThreshold', then the 'getNullSpaceStepByStep' command is used. If it has more rows that this parameter, then the external program SpaSM is called. ";
globalSpaSMThreshold=1000;

globalGetNullSpaceStep::usage=" This is a global parameter in the command 'getNullSpace'. When 'GetNullSpace' uses the 'getNullSpaceStepByStep' algorithm, the matrix whose null space one wants to compute is divided into submatrices of row size given by 'globalGetNullSpaceStep'. ";
globalGetNullSpaceStep=200;

globalGetNullSpaceSpaSMPrimes::usage=" This is a global parameter in the command 'getNullSpace'. It is the list of prime numbers that are given to SpaSM when 'getNullSpace' calls the command FFRREF. ";
globalGetNullSpaceSpaSMPrimes=Take[globalSpaSMListOfPrimes,-4];



(* ::Section::Initialization:: *)
(*(*Global parameters of 'rowReduceMatrix' and 'rowReduceOverPrimes'*)*)


(* ::Input::Initialization:: *)
globalSetOfBigPrimes::usage=" This is a global parameter in the command 'rowReduceOverPrimes'. It is a list of very big primes that are used in performing a row reduction over a dense matrix.  ";
globalSetOfBigPrimes=Select[2^63-Range[983],PrimeQ];

globalRowReduceOverPrimesInitialNumberOfIterations::usage=" This is a global parameter in the command 'rowReduceOverPrimes'. It specifies the initial number of row reductions over prime numbers that the command makes before it start reconstructing the row reduced matries over larger numbers by using the Chinese remained algorithm.  ";
globalRowReduceOverPrimesInitialNumberOfIterations=2;


globalRowReduceOverPrimesMaxNumberOfIterations::usage=" This is a global parameter in the command 'rowReduceOverPrimes'. It specifies the maximal number of row reductions over prime numbers that the command is allowed to make.   ";
globalRowReduceOverPrimesMaxNumberOfIterations=10;

globalRowReduceOverPrimesMethod::usage=" This parameter set the way the command 'rowReduceOverPrimes' chooses its primes from the list 'globalSetOfBigPrimes'. If the value is 'Systematic', then the command chooses the first 'globalRowReduceOverPrimesMaxNumberOfIterations' elements of 'globalSetOfBigPrimes' as its primes. If the value is 'Random', then the primes are randomly chosen.  ";
globalRowReduceOverPrimesMethod="Systematic"; 



globalRowReduceMatrixSpaSMPrimes::usage=" This is a global parameter in the command 'rowReduceMatrix'. It specifies the primes that are used when calling the external program SpaSM.";
globalRowReduceMatrixSpaSMPrimes=Take[globalSpaSMListOfPrimes,-4];


(* ::Section::Initialization:: *)
(*(*Resetting the global parameters/choosing various prepackages possibilities*)*)


(* ::Input::Initialization:: *)
resetTheGlobalParameters[]:=Module[{},
globalLowerThreshold=200; 
globalSpaSMThreshold=1000;
globalGetNullSpaceStep=200;
globalSpaSMListOfPrimes=Select[Range[2^14]+10000,PrimeQ];
globalGetNullSpaceSpaSMPrimes=Take[globalSpaSMListOfPrimes,-4];
globalSetOfBigPrimes=Select[2^63-Range[983],PrimeQ];
globalRowReduceOverPrimesInitialNumberOfIterations=2;
globalRowReduceOverPrimesMaxNumberOfIterations=10;
globalRowReduceOverPrimesMethod="Random"; 
globalRowReduceMatrixSpaSMPrimes=Take[globalSpaSMListOfPrimes,-4];
globalSpaSMSwitch=False;
globalSymBuildParallelize=False;
Return["The global variables have been reset to their standard values. "] 
];


(* ::Input::Initialization:: *)



(* ::Title::Initialization:: *)
(*(*The private part of the package*)*)


(* ::Section:: *)
(*Beginning*)


Begin["`Private`"] (* Begin Private Context *)


(* ::Subsubsection:: *)
(*General commands on lists and matrices manipulations*)


(*Sparse Matrix Manipulation*)
sparseArrayGlueRight::nnarg=" The dimensions of the matrices are mismatched! ";

sparseArrayGlueRight[A1_,A2_]/;If[Dimensions[A1][[1]]== Dimensions[A2][[1]],True,Message[sparseArrayGlueRight::nnarg];False]:=SparseArray[Union[A1//ArrayRules,(A2//ArrayRules)/.{a1_,a2_}:> {a1,a2+Dimensions[A1][[2]]}/;!(a1===_)],{Dimensions[A1][[1]],Dimensions[A1][[2]]+Dimensions[A2][[2]]}];

sparseArrayGlue::nnarg=" The dimensions of the matrices are mismatched! ";

sparseArrayGlue[A1_,A2_]/;If[Dimensions[A1][[2]]== Dimensions[A2][[2]],True,Message[sparseArrayGlue::nnarg];False]:=SparseArray[Union[A1//ArrayRules,(A2//ArrayRules)/.{a1_,a2_}:> {a1+Dimensions[A1][[1]],a2}/;!(a1===_)],{Dimensions[A1][[1]]+Dimensions[A2][[1]],Dimensions[A1][[2]]}];

sparseArrayGlue[A_]:=Which[Length[A]==0,A,Length[A]==1,A[[1]], Length[A]==2,sparseArrayGlue[A[[1]],A[[2]]],Length[A]>2, sparseArrayGlue[Join[{sparseArrayGlue[A[[1]],A[[2]]]},Drop[A,2]]]];

sparseArrayZeroRowCut[sarray_]:=Module[{entriesPosition=Drop[(sarray//ArrayRules)[[All,1]],-1],dimArray=Dimensions[sarray]},If[entriesPosition==={},SparseArray[sarray,{1,dimArray[[2]]}],SparseArray[sarray,{Max[entriesPosition[[All,1]]],dimArray[[2]]}]]];

(*---------------------------------------------------------------------*)

dotSymbolTensors[tensorL_,tensorR_]:=tensorL.tensorR;
dotSymbolTensors[A_]:=Which[Length[A]==0,{},Length[A]==1,A[[1]], Length[A]==2,dotSymbolTensors[A[[1]],A[[2]]],Length[A]>2, dotSymbolTensors[Join[{dotSymbolTensors[A[[1]],A[[2]]]},Drop[A,2]]]];

(*---------------------------------------------------------------------*)

transposeLevelsSparseArray[sparseArray_,level1_,level2_]:=Module[{array=Drop[sparseArray//ArrayRules,-1], dimensions=Dimensions[sparseArray],swappedOrder},
swappedOrder=Range[Length[dimensions]]/.{level1-> level2,level2-> level1};
SparseArray[Table[array[[iter,1]][[swappedOrder]]-> array[[iter,2]],{iter,1,Length[array]}],dimensions[[swappedOrder]]]
];

transposeLevelsSparseArray[sparseArray_,swappedOrder_]:=Module[{array=Drop[sparseArray//ArrayRules,-1], dimensions=Dimensions[sparseArray]},
SparseArray[Table[array[[iter,1]][[swappedOrder]]-> array[[iter,2]],{iter,1,Length[array]}],dimensions[[swappedOrder]]]
];

positionDuplicates[list_]:=GatherBy[Range@Length[list],list[[#]]&] ;



(* Dense Matrix Manipulation*)

denseMatrixConcatenateBelow::nnarg=" The dimensions of the matrices are mismatched! ";

denseMatrixConcatenateBelow[matrixUp_,matrixDown_]/;If[Dimensions[matrixUp][[2]]== Dimensions[matrixDown][[2]],True,Message[denseMatrixConcatenateBelow::nnarg];False]:=ArrayFlatten[{{matrixUp},{matrixDown}}];

denseMatrixConcatenateRight::nnarg=" The dimensions of the matrices are mismatched ";

denseMatrixConcatenateRight[matrixLeft_,matrixRight_]/;If[Dimensions[matrixLeft][[1]]== Dimensions[matrixRight][[1]],True,Message[denseMatrixConcatenateRight::nnarg];False]:=ArrayFlatten[{{matrixLeft,matrixRight}}];

denseMatrixZeroRowCut[matrix_]:=DeleteCases[#,ConstantArray[0,Length@#[[1]]]]&@matrix;


(* Finding a set of linear independent columns of a matrix *)
linearIndependentColumns[mat_]:=Map[Position[#,Except[0,_?NumericQ],1,1]&,RowReduce[mat]]//Flatten;


(*---------------------------------------------------------------------*)
(* UPDATE NEEDED: THE ROW REDUCTION SHOULD NOT BE NECESSARY SINCE WE WANT A VERY SPECIAL CASE! *)

determineLeftInverse::nnarg=" The number of rows must be bigger or equal to the number of columns! ";

determineLeftInverse[sparseMatrix_]/;If[Dimensions[sparseMatrix][[1]]>= Dimensions[sparseMatrix][[2]],True,Message[determineLeftInverse::nnarg];False]:=Module[{rowLength, columnLength, TEMPmatrix},
{rowLength, columnLength}=Dimensions[sparseMatrix]; 
TEMPmatrix=rowReduceMatrix[sparseArrayGlueRight[sparseMatrix,SparseArray[Band[{1,1}]-> 1,{rowLength,rowLength}]]];
Return[SparseArray[TEMPmatrix[[1;;columnLength, columnLength+1;;]]]];
];


(* ::Subsubsection:: *)
(*Formal symbols manipulations*)


(* ::Input::Initialization:: *)



shuffles[A1_,A2_]:=Module[{nfoobar,p1,p2,shuffledz,A12},nfoobar=Length/@{A1,A2};
{p1,p2}=Subsets[Range@Tr@nfoobar,{#}]&/@nfoobar;
p2=Reverse@p2;
A12=shuffledz=Join[A1,A2];
(shuffledz[[#]]=A12;shuffledz)&/@Join[p1,p2,2]];




(* ::Subsubsection:: *)
(*Miscellaneous*)


modifiedRowReduce[sparseArray_]:=RowReduce[Normal[sparseArray]];



(* ::Subsubsection::Initialization:: *)
(*(*Rational reconstruction*)*)


(* ::Input::Initialization:: *)
rationalReconstructionAlgorithm[q_,prime_]:=Module[{r0=q, s0=1, r1=prime,s1=0,rnew,snew,qnew},
If[q==0,Return[q]];
While[prime-r1^2<  0,
 qnew=Floor[r0/r1]; rnew=r0-qnew r1; snew=s0-qnew s1;
{r0,s0}={r1,s1};
{r1,s1}={rnew,snew};];
Return[r1/s1]
];

rationalReconstructionArray[array_,prime_]:=Module[{TEMPArray=SparseArray[array]},Map[rationalReconstructionAlgorithm[#,prime]&,TEMPArray,{Depth[TEMPArray]-1}]];


(* ::Subsubsection:: *)
(*Row Reduction over the finite fields for a dense matrix*)


applyChineseRemainder[matrixList_,primesList_]:=Module[{listOfEntries=Union[Flatten[(Most[ArrayRules[#1][[All,1]]]&)/@matrixList,1]]},
SparseArray[Table[foo->ChineseRemainder[Table[matrixFoo[[Sequence@@foo]],{matrixFoo,matrixList}],primesList],{foo,listOfEntries}],Dimensions[First[matrixList]]]
];




rowReduceOverPrimes[matrix_]:=Module[{samplePrimes,primeList,reducedMatrix,reducedMatrixReconstructed,
TEMPlistOfBigPrimes=globalSetOfBigPrimes,TEMPmatrixList,TEMPprimeList,tallyList,iterbar},
Which[globalRowReduceOverPrimesMethod=="Systematic", samplePrimes=Take[TEMPlistOfBigPrimes,globalRowReduceOverPrimesMaxNumberOfIterations];,
globalRowReduceOverPrimesMethod=="Random",  samplePrimes=RandomSample[TEMPlistOfBigPrimes,globalRowReduceOverPrimesMaxNumberOfIterations];,
True,Return["Error, the variable 'globalRowReduceOverPrimesMethod' should be either 'Systematic' or 'Random'!" ]];
If[globalSymBuildParallelize,
(*Parallel*)
DistributeDefinitions[samplePrimes,globalRowReduceOverPrimesInitialNumberOfIterations];
reducedMatrix=ParallelTable[RowReduce[matrix,Modulus->samplePrimes[[iterfoo]]],{iterfoo,1,globalRowReduceOverPrimesInitialNumberOfIterations}];
DistributeDefinitions[reducedMatrix];
reducedMatrixReconstructed=ParallelTable[rationalReconstructionArray[reducedMatrix[[iterfoo]],samplePrimes[[iterfoo]]],{iterfoo,1,globalRowReduceOverPrimesInitialNumberOfIterations}];
,
(*Series*)
reducedMatrix=Table[RowReduce[matrix,Modulus->samplePrimes[[iterfoo]]],{iterfoo,1,globalRowReduceOverPrimesInitialNumberOfIterations}];
reducedMatrixReconstructed=Table[rationalReconstructionArray[reducedMatrix[[iterfoo]],samplePrimes[[iterfoo]]],{iterfoo,1,globalRowReduceOverPrimesInitialNumberOfIterations}];
];
tallyList=Tally[reducedMatrixReconstructed];
If[tallyList[[1,2]]/globalRowReduceOverPrimesInitialNumberOfIterations>1/2,Return[tallyList[[1,1]]]];
(*Print[MatrixForm[#]&/@reducedMatrixReconstructed];*)
For[iterbar=globalRowReduceOverPrimesInitialNumberOfIterations+1,iterbar< globalRowReduceOverPrimesMaxNumberOfIterations+1,iterbar++,
PrintTemporary["Need more than " <>ToString[globalRowReduceOverPrimesInitialNumberOfIterations]<>" primes. Trying with "<>ToString[iterbar]<>" primes."];
reducedMatrix=Append[reducedMatrix,RowReduce[matrix,Modulus->samplePrimes[[iterbar]]]];
primeList=Take[samplePrimes,iterbar];
(*The 'SparseArray' part in the table below removes rare instances of sparse matrices in which entries of the type {a,b}\[Rule] 0 have been saved separately. *)
reducedMatrixReconstructed=Table[TEMPmatrixList=Drop[reducedMatrix,{iter}];TEMPprimeList=Drop[primeList,{iter}];
SparseArray[rationalReconstructionArray[applyChineseRemainder[TEMPmatrixList,TEMPprimeList],Times@@TEMPprimeList]]
,{iter,1,Length[reducedMatrix]}];
tallyList=Tally[reducedMatrixReconstructed];
If[tallyList[[1,2]]/iterbar>1/2,Return[tallyList[[1,1]]]];
];
Return["No solution! Choose different primes or increase the iteration! "];
];


(*
If[globalSymBuildParallelize,
(*Parallel*)
DistributeDefinitions[resolveRootViaGroebnerBasis,arrayToSimplify,listOfRoots,listOfRootPowers];
*)


(* ::Subsubsection:: *)
(*Null space commands*)


getNullSpaceFromRowReducedMatrix[rowReducedMatrix_]:=Block[{freeCoeff,dependentCoeff,trivialIntgrb,solutions,matrixNumberOfColumns=Dimensions[rowReducedMatrix][[2]],sortedEntries},
sortedEntries=GatherBy[Map[First,Most[ArrayRules[rowReducedMatrix]]],First[#]&];
freeCoeff=Union[Last/@Flatten[Rest/@sortedEntries,1]];dependentCoeff=Map[Last[First[#]]&,sortedEntries];trivialIntgrb=Map[{#->1}&,Complement[Range[matrixNumberOfColumns],dependentCoeff,freeCoeff]];solutions=Map[Map[dependentCoeff[[First[First[#]]]]-> Last[#]&,Most[ArrayRules[rowReducedMatrix.SparseArray[#->-1,{matrixNumberOfColumns}]]]]&,freeCoeff];solutions=MapThread[Join,{solutions,Map[{#->1}&,freeCoeff]}];solutions=Join[trivialIntgrb,solutions];
SparseArray[Map[SparseArray[#,{matrixNumberOfColumns}]&, solutions],{Length[solutions],Dimensions[rowReducedMatrix][[2]]}]
]; 


(*---------------------------------------------------------------------*)

getNullSpaceStepByStep::nnarg=" The variable 'step' should be smaller than the number of rows in the matrix! ";

getNullSpaceStepByStep[matrix_,step_]/;If[Dimensions[matrix][[1]]>= step,True,Message[getNullSpace::nnarg];False]:=Module[
{outputMonitoring="Preparing to compute the null space.",n0,numberOfIterations,
TEMPmatrix,oldRank,newRank,lenMatrix=First[Dimensions[matrix]],checkn0},
numberOfIterations=IntegerPart[lenMatrix/step]-1;
PrintTemporary["The number of full steps for the row reduction is "<>ToString[Ceiling[lenMatrix/step]]];
Monitor[
n0=getNullSpaceFromRowReducedMatrix[SparseArray[modifiedRowReduce[Take[matrix,step]]]];
oldRank=First[Dimensions[n0]];
Do[
TEMPmatrix=SparseArray[modifiedRowReduce[Take[matrix,{step j+1,step(j+1)}].Transpose[n0]]]; 
checkn0=getNullSpaceFromRowReducedMatrix[TEMPmatrix];
If[checkn0==={},Return[{}],n0=checkn0.n0];
newRank=First[Dimensions[n0]];If[oldRank<newRank,Print["Error in getting the null space"]];oldRank=newRank;
outputMonitoring={"Current step: "<>ToString[j+1],"Current dimensions of the null space: "<>ToString[Length[n0]],"Density of the sparse array: "<>ToString[n0["Density"]]};,
{j,1,numberOfIterations}];
If[Length[matrix]>step(numberOfIterations+1),
TEMPmatrix=SparseArray[modifiedRowReduce[Take[matrix,{step(numberOfIterations+1)+1,lenMatrix}].Transpose[n0]]];
checkn0=getNullSpaceFromRowReducedMatrix[TEMPmatrix];
If[checkn0==={},Return[{}],n0=checkn0.n0];
newRank=First[Dimensions[n0]];
If[oldRank<newRank,Print["Error in getting the null space!"]];
outputMonitoring={"Current step: "<>ToString[numberOfIterations+1],"Current dimensions of the null space: "<>ToString[Length[n0]],"Density of the sparse array: "<>ToString[n0["Density"]]};
Return[n0],
Return[n0]];
,outputMonitoring]
];


(* ::Subsubsection:: *)
(*Two auxiliary commands for the determination of transformation matrices between two alphabets*)


auxFlattenTwoIndices12[sparsearray_,sizeAlphabet_]:=SparseArray[Most[sparsearray//ArrayRules]/. ({a1_,a2_,a3_}->a4_):>({(a1-1) sizeAlphabet+a2,a3}->a4),{Dimensions[sparsearray][[1]]*sizeAlphabet,Dimensions[sparsearray][[3]]}];


auxFlattenTwoIndices23[sparsearray_,sizeAlphabet_]:=SparseArray[Most[sparsearray//ArrayRules]/. ({a1_,a2_,a3_}->a4_):>({a1,(a2-1) sizeAlphabet+a3}->a4),{Dimensions[sparsearray][[1]],Dimensions[sparsearray][[2]]*sizeAlphabet}];


(* ::Subsubsection:: *)
(*Presentation commands*)


presentIntegrableSymbolsData[tensorList_]:=Print["----------------------------------------- \n", Dimensions[tensorList][[3]]," integrable symbols in an alphabet with ",Dimensions[tensorList][[2]]," letters. The number of symbols of previous weight is ",Dimensions[tensorList][[1]],". \n-----------------------------------------" ];

presentIntegrableSymbolsData[{tensorList_,signsArray_}]:=Print["----------------------------------------- \n", Dimensions[tensorList][[3]]," integrable symbols (",Count[signsArray,0]," even, ",Count[signsArray,1]," odd) in an alphabet with ",Dimensions[tensorList][[2]]," letters. The number of symbols of previous weight is ",Dimensions[tensorList][[1]],". \n-----------------------------------------" ];


(* ::Subsubsection::Initialization:: *)
(*(*Commands for the computation of Even + odd symbols *)*)
(*(**)*)


makeSparseMatrixOutOfIndexLists[index1_,index2_,size1_,size2_]:=Module[{biIndexTable=Flatten[Table[{fooH,fooL},{fooH,index1},{fooL,index2}],1]},
SparseArray[Table[{iter,(biIndexTable[[iter,1]]-1)size2+biIndexTable[[iter,2]]}-> 1,{iter,1,Length[biIndexTable]}]//Flatten,{Length[biIndexTable], size1 size2 }]];

makeTheEvenOddConditionsMatrix[previousIntegrableSymbolsSigns_,listOfSymbolSigns_,evenOrOdd_]:=Module[{even1,odd1,even2,odd2,mat1,mat2},
even1=Position[previousIntegrableSymbolsSigns,0]//Flatten;
odd1=Position[previousIntegrableSymbolsSigns,1]//Flatten;
even2=Position[listOfSymbolSigns,0]//Flatten;
odd2=Position[listOfSymbolSigns,1]//Flatten;
If[evenOrOdd==0,
mat1=makeSparseMatrixOutOfIndexLists[even1,odd2,Length[previousIntegrableSymbolsSigns],Length[listOfSymbolSigns]];
mat2=makeSparseMatrixOutOfIndexLists[odd1,even2,Length[previousIntegrableSymbolsSigns],Length[listOfSymbolSigns]];
Which[mat1==={}&&mat2==={},Return[SparseArray[{Table[0,Length[previousIntegrableSymbolsSigns]*Length[listOfSymbolSigns]]}]];,
mat1==={},Return[mat2];,
mat2==={},Return[mat1];,
True,Return[sparseArrayGlue[mat1,mat2]]];
,
mat1=makeSparseMatrixOutOfIndexLists[even1,even2,Length[previousIntegrableSymbolsSigns],Length[listOfSymbolSigns]];
mat2=makeSparseMatrixOutOfIndexLists[odd1,odd2,Length[previousIntegrableSymbolsSigns],Length[listOfSymbolSigns]];
Which[mat1==={}&&mat2==={},Return[SparseArray[{Table[0,Length[previousIntegrableSymbolsSigns]*Length[listOfSymbolSigns]]}]];,
mat1==={},Return[mat2];,
mat2==={},Return[mat1];,
True,Return[sparseArrayGlue[mat1,mat2]]];
];
];



(* ::Section:: *)
(*End*)


End[] (* End Private Context *)


(* ::Title::Initialization:: *)
(*(*The public part of the package*)*)


(* ::Chapter::Initialization::Closed:: *)
(*(*Symbol tensors and their manipulation*)*)


(* ::Section::Initialization::Closed:: *)
(*(*Glue two lists of tensors that give integrable symbols*)*)


(* ::Input::Initialization:: *)
integrableSymbolsTensorsGlue::nnarg=" The dimensions of the tensors are mismatched! ";

integrableSymbolsTensorsGlue[A1_,A2_]/;If[Dimensions[A1][[1]]== Dimensions[A2][[1]]&&Dimensions[A1][[2]]== Dimensions[A2][[2]],True,Message[integrableSymbolsTensorsGlue::nnarg];False]:=Module[{dim=Dimensions[A1][[3]]},
SparseArray[Union[ArrayRules[A1],(ArrayRules[A2]/. {a1_,a2_,a3_}:>{a1,a2,a3+dim}/;!a1===_)],{Dimensions[A1][[1]],Dimensions[A1][[2]],Dimensions[A1][[3]]+Dimensions[A2][[3]]}]
];


(* ::Section::Initialization:: *)
(*(*Writing the null spaces into tensors and doing the reverse*)*)


(* ::Input::Initialization:: *)
solutionSpaceToSymbolsTensor::nnarg=" The dimensions of 'sparseSolutionSpace' do not match the size of the alphabet.";

solutionSpaceToSymbolsTensor[sparseSolutionSpace_,sizeAlphabet_]/;If[Mod[Dimensions[sparseSolutionSpace][[2]],sizeAlphabet]==0,True,Message[solutionSpaceToSymbolsTensor::nnarg];False]:=Module[{previousSolSpaceLenth},previousSolSpaceLenth=Dimensions[sparseSolutionSpace][[2]]/sizeAlphabet;
SparseArray[Most[sparseSolutionSpace//ArrayRules]/.Rule[{a1_,a2_},a3_]:>Rule[{Quotient[a2-1,sizeAlphabet]+1,Mod[a2,sizeAlphabet,1],a1},a3],{previousSolSpaceLenth,sizeAlphabet,Length[sparseSolutionSpace]}]
];

symbolsTensorToVector[symbolsTensor_]:=Module[{sizeAlphabet=Dimensions[symbolsTensor][[2]]},
SparseArray[Most[ArrayRules[symbolsTensor]]/. ({a1_,a2_}->a4_):>{(a1-1) sizeAlphabet+a2}->a4,{Dimensions[symbolsTensor][[1]]sizeAlphabet}]
];


(*---------------------------------------------------------------------*)


symbolsTensorToSolutionSpace::nnarg=" The argument of 'symbolsTensorToSolutionSpace' must be a tensor with 3 indices! (i.e. the depth should be equal to 4). ";

symbolsTensorToSolutionSpace[symbolsTensor_]/;If[Depth[symbolsTensor]==4,True,Message[symbolsTensorToSolutionSpace::nnarg];False]:=Module[{sizeAlphabet=Dimensions[symbolsTensor][[2]]},
SparseArray[Most[ArrayRules[symbolsTensor]]/. ({a1_,a2_,a3_}->a4_):>{a3,(a1-1) sizeAlphabet+a2}->a4,{Dimensions[symbolsTensor][[3]],Dimensions[symbolsTensor][[1]] sizeAlphabet}]
];


(* ::Section::Initialization:: *)
(*(*Rewriting the tensors into sums of formal symbols*)*)


(* ::Input::Initialization:: *)
(*---------------------------------------------------------------------*)

Default[expressTensorAsSymbols]="Complete";

expressTensorAsSymbols[sparseArrayRules_,options_.]:=Module[{tensorLength,positionTable,TEMPsparseTensor},
TEMPsparseTensor=If[Head[sparseArrayRules]===SparseArray,sparseArrayRules//ArrayRules,sparseArrayRules];
tensorLength=TEMPsparseTensor[[1,1]]//Length;
positionTable=positionDuplicates[Drop[TEMPsparseTensor[[All,1]][[All,tensorLength]],-1]];
Which[options=="Complete",
Table[Sum[TEMPsparseTensor[[ifoo,2]]sb[Drop[TEMPsparseTensor[[ifoo,1]],-1]],{ifoo,posList}],{posList,positionTable}],
options=="DropFirst",
Table[Sum[TEMPsparseTensor[[ifoo,2]]sb[Drop[Drop[TEMPsparseTensor[[ifoo,1]],-1],1]],{ifoo,posList}],{posList,positionTable}],
options=="Recursive",
Table[Sum[TEMPsparseTensor[[ifoo,2]]sb[\[ScriptCapitalS][TEMPsparseTensor[[ifoo,1]]//First],Drop[Drop[TEMPsparseTensor[[ifoo,1]],-1],1]],{ifoo,posList}],{posList,positionTable}]
]
];


(* ::Chapter::Initialization:: *)
(*(*Formal Symbols and their manipulations *)*)


(* ::Subsubsection::Initialization:: *)
(*(*Shuffle products*)*)


shuffleProduct[symbol1_,symbol2_]:=(symbol1 (symbol2/.SB[A_]:> SB2[A])//Expand)/.SB[A_]SB2[B_]:> Sum[SB[foo],{foo,shuffles[A,B]}];


(* ::Subsubsection::Initialization:: *)
(*(*Define the formal symbols*)*)


(* ::Input::Initialization:: *)
SB[{}]:=1;
SB[{A1___,A_,A2___}]:=0/;NumericQ[A];
SB[{A___,A1_^n_,B___}]:=n SB[{A,A1,B}];
SB[{A___,A1_ A2_,B___}]:=SB[{A,A1,B}]+SB[{A,A2,B}];
SB[{A___,A1_/A2_,B___}]:=SB[{A,A1,B}]-SB[{A,A2,B}];


(* ::Subsubsection::Initialization:: *)
(*(*Extracting limits of formal symbols*)*)


(* ::Input::Initialization:: *)
listifySum[A_]:=If[Head[A]===Plus,List@@A,{A}];


factorSymbols[symbolsExpression_]:=symbolsExpression/.SB[A_]:> SB[Factor/@A]/.SB[A_]:> SB[FactorList[#][[2,1]]&/@A];


extractSingularPart[symbol_,var_]:=Module[{listsymbol=listifySum[Expand[symbol]],logs},logs=Cases[listsymbol,SB[{A___,var,B___}],Infinity]//DeleteDuplicates;Sum[Coefficient[symbol,currentlog]currentlog,{currentlog,logs}]];


takeLimitOfSymbol[symbol_,var_]:=Module[{symbolSimple=Expand[Factor[factorSymbols[symbol]]],singularPart},
singularPart=extractSingularPart[symbolSimple,var];factorSymbols[((symbolSimple-singularPart//Expand)/.var-> 0)+singularPart/.SB[A_]:> SB[Table[If[(afoo/.var-> 0)===0,afoo,afoo/.var-> 0],{afoo,A}]]]
];


(* ::Subsubsection::Initialization:: *)
(*(*Expanding formal symbols in a given basis*)*)


(* ::Input::Initialization:: *)



expandInSymbolBasis[exp_,basis_]:=Module[{coefficientsTEMP,ansatzTEMP,solTEMP,expTEMP=factorSymbols[exp],basisTEMP=factorSymbols[basis],varSBTEMP},
varSBTEMP=Cases[basisTEMP,SB[___],Infinity]//DeleteDuplicates;
ansatzTEMP=Sum[coefficientsTEMP[iter] basisTEMP[[iter]],{iter,1,Length[basisTEMP]}];solTEMP=Quiet[Solve[0==(CoefficientArrays[expTEMP-ansatzTEMP,varSBTEMP]//Flatten//DeleteDuplicates),Table[coefficientsTEMP[i],{i,1,Length[basis]}]]];Which[solTEMP=={},"No solution.",
Cases[solTEMP[[1]][[All,2]],coefficientsTEMP[___],Infinity]==={},Table[coefficientsTEMP[i],{i,1,Length[basis]}]/. solTEMP[[1]],
True,"The supposed basis is not linearly independent."]
];


(* ::Subsubsection::Initialization:: *)
(*(*Projecting products away*)*)


(* ::Input::Initialization:: *)
subProductProjection[symbol_]:=Module[{list=(symbol/.SB[A_]:> A),lengthList},
lengthList=Length[list]; 
If[lengthList==1,Return[symbol]]; 
(lengthList-1)/lengthList ((subProductProjection[SB[Drop[list,-1]]]/.SB[A_]:> SB[Join[A,{list[[lengthList]]}]])-(subProductProjection[SB[Drop[list,1]]]/.SB[A_]:> SB[Join[A,{list[[1]]}]]))
];


productProjection[symbolExpression_]:=symbolExpression/.SB[A_]:> subProductProjection[SB[A]];


(* ::Subsubsection:: *)
(*Conversion of formal symbols*)


(*---------------------------------------------------------------------------*)
(* convert a formal lowercase symbol sb to an explicit formal symbol SB *)

convertFormalSymbol[expr_,alphabet_]:=expr/.sb[A_]:> If[Head[A]===List,SB[Table[alphabet[[foo]],{foo,A}]],sb[A]];



(* ::Subsubsection:: *)
(*Checking the integrability of a formal symbol*)



IntegrableQ::argerr="This can take a while! It is recommended to use the formal sb symbols when checking the integrability condition.";



IntegrableQ[expression_,FtensorOrVariables_]:=Module[{sbCases=Cases[{expression},sb[___],Infinity],SBCases=Cases[{expression},SB[___],Infinity]},
Which[sbCases==={}&& SBCases==={},
Return["IntegrableQ[expression_,FtensorOrAlphabet_] has to act on expressions containing sb[list_] or sb[\[ScriptCapitalS][i],list_], where 'list' has at least 2 elements. "],
SBCases==={},
sbCases=DeleteDuplicates[Length/@(sbCases/.sb[\[ScriptCapitalS][x_],A_]:> A/.sb[A_]:> A)];
If[!((FtensorOrVariables//Dimensions//Length)===3),Return["When using sb formal symbols, the second entry of IntegrableQ[expression_,FtensorOrAlphabet_] must be an integrability tensor!"]];
If[
Length[sbCases]>1,
Return["IntegrableQ[expression_,FtensorOrAlphabet_] has to act on expressions containing sb[list_] or sb[\[ScriptCapitalS][i],list_], where 'list' has at least 2 elements. Furthermore, the formal symbols have to be expressed uniformly"],
Return[{0}===((expression/.sb[\[ScriptCapitalS][x_],A_]:> Table[sb[\[ScriptCapitalS][x],Join[A[[1;;(iter-1)]],A[[(iter+2);;]]]]FtensorOrVariables[[All,A[[iter]],A[[iter+1]]]],{iter,1,Length[A]-1}]/.sb[A_]:> Table[sb[Join[A[[1;;(iter-1)]],A[[(iter+2);;]]]]FtensorOrVariables[[All,A[[iter]],A[[iter+1]]]],{iter,1,Length[A]-1}])//Flatten//DeleteDuplicates)];
];,
sbCases==={},
If[!((FtensorOrVariables//Dimensions//Length)===1),Return["When using the SB formal symbols, the second entry of IntegrableQ[expression_,FtensorOrAlphabet_] must be a list of variables!"]];Message[IntegrableQ::argerr];
Return[{0}===((expression/.SB[A_]:> Table[SB[Join[A[[1;;(iter-1)]],A[[(iter+2);;]]]]Table[Factor[D[Log[A[[iter]]],FtensorOrVariables[[vara]]] D[Log[A[[iter+1]]],FtensorOrVariables[[varb]]]-D[Log[A[[iter]]],FtensorOrVariables[[varb]]] D[Log[A[[iter+1]]],FtensorOrVariables[[vara]]]],{vara,1,Length[FtensorOrVariables]},{varb,vara+1,Length[FtensorOrVariables]}],{iter,1,Length[A]-1}])//Factor//Flatten//DeleteDuplicates)];,
True,
Return["IntegrableQ[expression_,FtensorOrAlphabet_] cannot mix both sb and SB formal symbols. "]
];
];


(* ::Chapter::Initialization::Closed:: *)
(*(*Null Space commands*)*)


getNullSpace[matrix_]:=Module[{nullMatrix},
Which[
Length[matrix]<globalLowerThreshold, 
nullMatrix=NullSpace[matrix]; If[nullMatrix==={},Return[{}],Return[SparseArray[NullSpace[matrix]]]],
(Length[matrix]>=globalSpaSMThreshold)&&globalSpaSMSwitch,
Return[getNullSpaceFromRowReducedMatrix[FFRREF[matrix,globalGetNullSpaceSpaSMPrimes,MatrixDirectory->globalSpaSMExchangePath,Nkernel->globalSpaSMNumberOfKernels]]],
True, Return[getNullSpaceStepByStep[matrix,globalGetNullSpaceStep]]];
];



(* ::Chapter::Initialization:: *)
(*(*Checking the independence of the alphabet*)*)


(* ::Input::Initialization:: *)
dLogAlphabet[alphabet_,allvariables_,listOfRoots_,listOfRootPowers_]:=Module[{TEMPeqn,newVariables,variablesRedef,variablesRedefReverse,TEMPalphabet,newRoots},

newVariables=Table[ToExpression["xTEMP"<>ToString[i]],{i,1,Length[allvariables]}];
variablesRedef=Table[allvariables[[i]]->newVariables[[i]],{i,1,Length[newVariables]}];
variablesRedefReverse=Table[newVariables[[i]]->allvariables[[i]],{i,1,Length[newVariables]}];
TEMPalphabet=alphabet/.variablesRedef;
newRoots=listOfRoots/.variablesRedef;

TEMPeqn=Table[D[Log[TEMPalphabet[[iterletter]]],newVariables[[itervar]]],{itervar,1,Length[newVariables]},{iterletter,1,Length[TEMPalphabet]}];

TEMPeqn=(TEMPeqn/.\!\(\*
TagBox[
StyleBox[
RowBox[{
RowBox[{
RowBox[{"Derivative", "[", "derInder__", "]"}], "[", 
RowBox[{"root", "[", "a_", "]"}], "]"}], "[", "X__", "]"}],
ShowSpecialCharacters->False,
ShowStringCharacters->True,
NumberMarks->True],
FullForm]\):> root[a]/(newRoots[[a]]listOfRootPowers[[a]]) Module[{tempList=List[derInder]},D[newRoots[[a]],Sequence@@Table[{newVariables[[i]],tempList[[i]]},{i,1,Length[tempList]}]]] );
TEMPeqn=TEMPeqn/.root[a_][X__]:> root[a]/.root[a_]:> ToExpression["\[Rho]"<>ToString[a]]/.variablesRedefReverse;
Return[TEMPeqn]
];



findRelationsInAlphabet[alphabet_,allvariables_,listOfRoots_,listOfRootPowers_,sampleSize_,maxSamplePoints_,toleranceForRetries_]:=Module[{TEMPdLogEquations,TEMPmatrix,TEMPNullSpace},

TEMPdLogEquations=dLogAlphabet[alphabet,allvariables,listOfRoots,listOfRootPowers];
TEMPdLogEquations=resolveRootViaGroebnerBasisMatrix[TEMPdLogEquations,listOfRoots,listOfRootPowers];
TEMPdLogEquations=collectRootCoefficients[TEMPdLogEquations,Table[ToExpression["\[Rho]"<>ToString[iter]],{iter,1,Length[listOfRoots]}]];

TEMPmatrix=buildFMatrixReducedForASetOfEquations[TEMPdLogEquations,allvariables,sampleSize,maxSamplePoints,toleranceForRetries];
TEMPNullSpace=getNullSpaceFromRowReducedMatrix[TEMPmatrix];
Return[If[TEMPNullSpace=={},"The alphabet is indepedent.", {"The alphabet is dependent with these linear relations:", TEMPNullSpace}]]
];


(* ::Chapter::Initialization::Closed:: *)
(*(*Difference equations/Counting products and irreducible symbols*)*)


(* ::Subsubsection::Initialization:: *)
(*(*Computing the difference equation if given a sequence of dimensions*)*)


(* ::Input::Initialization:: *)



computeCoefficientsOfDifferenceEquation[dimSequence_]:=Module[{M=dimSequence[[2]],Rsequence,TEMPeqn,varAlpha,tempDim,tempCond,tempa},
Rsequence=Table[M dimSequence[[i-1]]-dimSequence[[i]],{i,2,Length[dimSequence]}];
TEMPeqn=Table[-Sum[(-1)^n tempa[n]tempDim[L-n],{n,1,L}]-(tempDim[L-1] M-tempCond[L]),{L,0,Length[dimSequence]-1}]/.{tempDim[-1]-> 0,tempCond[0]-> 0,tempCond[1]-> 0,tempDim[0]-> 1};
TEMPeqn=TEMPeqn/.{tempDim[a_]:>dimSequence[[a+1]],tempCond[a_]:> Rsequence[[a]]};
varAlpha=Cases[TEMPeqn,tempa[_],Infinity]//DeleteDuplicates;
Return[Table[tempa[i],{i,0,Length[varAlpha]}]/.Solve[TEMPeqn==0,varAlpha][[1]]/.tempa[0]-> 1]
];



(* ::Subsubsection::Initialization:: *)
(*(*Counting the number of products and of irreducible symbols*)*)


(* ::Input::Initialization:: *)



rewritePartition[partition_]:=Module[{max=Max[partition]},Table[Count[partition,i],{i,1,max}]];


dimProductSymbols[L_]:=Module[{tempPartitions},
If[L==0,Return[0]];
tempPartitions=Drop[(rewritePartition[#]&/@IntegerPartitions[L]),1];
Sum[Product[Binomial[dimQ[jfoo]+fooPartition[[jfoo]]-1,fooPartition[[jfoo]]],{jfoo,1,Length[fooPartition]}],{fooPartition,tempPartitions}]
];



dimIrreducibleSymbols[cutoffWeight_]:=
Table[dimQ[weight],{weight,0,cutoffWeight}]/.Solve[Table[dimQ[weight]- (dimH[weight]- FunctionExpand[dimProductSymbols[weight]]),{weight,0,cutoffWeight}]==0,Table[dimQ[weight],{weight,0,cutoffWeight}]][[1]];


(* ::Chapter::Initialization:: *)
(*(*Rational reconstruction algorithms*)*)


(* ::Chapter::Initialization:: *)
(*(*Row reduction (over the finite fields)*)*)


(* ::Section::Initialization:: *)
(*(*The general row reduction command*)*)


rowReduceMatrix[matrix_]:=Which[Length[matrix]<globalLowerThreshold,SparseArray[RowReduce[Normal[matrix]]],
(Length[matrix]>=globalSpaSMThreshold)&&globalSpaSMSwitch,FFRREF[matrix,globalRowReduceMatrixSpaSMPrimes,MatrixDirectory->globalSpaSMExchangePath,Nkernel->globalSpaSMNumberOfKernels],
True,rowReduceOverPrimes[matrix]
];



(* ::Chapter::Initialization:: *)
(*(*Computing the integrability tensor \[DoubleStruckCapitalF]*)*)


(* ::Section::Initialization::Closed:: *)
(*(*Transforming the reduced \[DoubleStruckCapitalM] matrix into the integrability tensor \[DoubleStruckCapitalF]*)*)


(* ::Input::Initialization:: *)
matrixFReducedToTensor[sparseMatrix_]:=Module[{TEMPdim=Dimensions[sparseMatrix],TEMPIndexTable,len},
len=(Sqrt[8TEMPdim[[2]]+1]+1)/2;
TEMPIndexTable=Flatten[Table[{i,j},{i,1,len-1},{j,i+1,len}],1];
SparseArray[Flatten[Table[{Join[{foo[[1,1]]},TEMPIndexTable[[foo[[1,2]]]]]-> foo[[2]],Join[{foo[[1,1]]},Reverse[TEMPIndexTable[[foo[[1,2]]]]]]-> -foo[[2]]},{foo,Drop[ArrayRules[sparseMatrix],-1]}]]
,{TEMPdim[[1]],len,len}]
];


(* ::Section::Initialization:: *)
(*(*Generating the set of equations involving only rational functions from which \[DoubleStruckCapitalF] is made*)*)


integrableEquationsRational[alphabet_,allvariables_]:=Module[{TEMPeqn,listOfIndices,newVariables,variablesRedef,variablesRedefReverse,TEMPalphabet,newRoots},

listOfIndices=Flatten[Table[{iter1,iter2},{iter1,1,Length[allvariables]-1},{iter2,iter1+1,Length[allvariables]}],1];

newVariables=Table[ToExpression["xTEMP"<>ToString[i]],{i,1,Length[allvariables]}];
variablesRedef=Table[allvariables[[i]]->newVariables[[i]],{i,1,Length[newVariables]}];
variablesRedefReverse=Table[newVariables[[i]]->allvariables[[i]],{i,1,Length[newVariables]}];
TEMPalphabet=alphabet/.variablesRedef;

TEMPeqn=Monitor[Table[Flatten[Table[D[Log[TEMPalphabet[[iter1]]],newVariables[[listOfIndices[[iterbaz]][[1]]]]]D[Log[TEMPalphabet[[iter2]]],newVariables[[listOfIndices[[iterbaz]][[2]]]]]-D[Log[TEMPalphabet[[iter1]]],newVariables[[listOfIndices[[iterbaz]][[2]]]]]D[Log[TEMPalphabet[[iter2]]],newVariables[[listOfIndices[[iterbaz]][[1]]]]],{iter1,1,Length[TEMPalphabet]-1},{iter2,iter1+1,Length[TEMPalphabet]}]],{iterbaz,1,Length[listOfIndices]}],iterbaz];

Return[TEMPeqn/.variablesRedefReverse]
];



integrableEquationsWithRoots[alphabet_,allvariables_,listOfRoots_,listOfRootPowers_]:=Module[{TEMPeqn,listOfIndices,newVariables,variablesRedef,variablesRedefReverse,TEMPalphabet,newRoots},
listOfIndices=Flatten[Table[{iter1,iter2},{iter1,1,Length[allvariables]-1},{iter2,iter1+1,Length[allvariables]}],1];

newVariables=Table[ToExpression["xTEMP"<>ToString[i]],{i,1,Length[allvariables]}];
variablesRedef=Table[allvariables[[i]]->newVariables[[i]],{i,1,Length[newVariables]}];
variablesRedefReverse=Table[newVariables[[i]]->allvariables[[i]],{i,1,Length[newVariables]}];
TEMPalphabet=alphabet/.variablesRedef;
newRoots=listOfRoots/.variablesRedef;

TEMPeqn=Monitor[Table[Flatten[Table[D[Log[TEMPalphabet[[iter1]]],newVariables[[listOfIndices[[iterbaz]][[1]]]]]D[Log[TEMPalphabet[[iter2]]],newVariables[[listOfIndices[[iterbaz]][[2]]]]]-D[Log[TEMPalphabet[[iter1]]],newVariables[[listOfIndices[[iterbaz]][[2]]]]]D[Log[TEMPalphabet[[iter2]]],newVariables[[listOfIndices[[iterbaz]][[1]]]]],{iter1,1,Length[TEMPalphabet]-1},{iter2,iter1+1,Length[TEMPalphabet]}]],{iterbaz,1,Length[listOfIndices]}],iterbaz];

(*Take the derivatives and express the derivatives of the roots nicely *)
TEMPeqn=(TEMPeqn/.\!\(\*
TagBox[
StyleBox[
RowBox[{
RowBox[{
RowBox[{"Derivative", "[", "derInder__", "]"}], "[", 
RowBox[{"root", "[", "a_", "]"}], "]"}], "[", "X__", "]"}],
ShowSpecialCharacters->False,
ShowStringCharacters->True,
NumberMarks->True],
FullForm]\):> root[a]/(newRoots[[a]]listOfRootPowers[[a]]) Module[{tempList=List[derInder]},D[newRoots[[a]],Sequence@@Table[{newVariables[[i]],tempList[[i]]},{i,1,Length[tempList]}]]] );
TEMPeqn=TEMPeqn/.root[a_][X__]:> root[a]/.root[a_]:> ToExpression["\[Rho]"<>ToString[a]]/.variablesRedefReverse;
Return[TEMPeqn]
];



(* ::Subsubsection::Initialization:: *)
(*(*Resolve the roots using Gr\[ODoubleDot]bner bases*)*)


resolveRootViaGroebnerBasis::nnarg="The size of 'listOfRoots' must match the size of 'listOfRootPowers'! ";

resolveRootViaGroebnerBasis[expressionToSimplify_,listOfRoots_,listOfRootPowers_]/;If[Length[listOfRoots]==Length[listOfRootPowers],True,Message[resolveRootViaGroebnerBasis::nnarg];False]:=
Module[{
TEMPExpression,TEMPgrobBasis,TEMPsol,TEMPgrobBasisTry,
listOfRootNames,listOfMinPolynomials,
Xbaz,positionOfTheLinearPolynomialInXbaz,RT,listOfMinPolynomialsResolved},

TEMPExpression=expressionToSimplify//Factor;

listOfRootNames=Table[ToExpression["\[Rho]"<>ToString[ibaz]],{ibaz,1,Length[listOfRoots]}];
(* Ignore entries that don't contain roots. Makes it faster... *)
If[Table[Cases[TEMPExpression,fooRoot,Infinity],{fooRoot,listOfRootNames}]//Flatten,Return[TEMPExpression]];

listOfMinPolynomials=Prepend[Table[listOfRootNames[[ibat]]^listOfRootPowers[[ibat]]-RT[ibat],{ibat,1,Length[listOfRoots]}],Xbaz Denominator[TEMPExpression]-Numerator[TEMPExpression]];
listOfMinPolynomialsResolved=listOfMinPolynomials/.RT[ifoo_]:> listOfRoots[[ifoo]];

TEMPgrobBasisTry=TimeConstrained[GroebnerBasis[listOfMinPolynomialsResolved,Prepend[listOfRootNames,Xbaz],CoefficientDomain->RationalFunctions],1];
If[TEMPgrobBasisTry===$Aborted,
TEMPgrobBasis=GroebnerBasis[listOfMinPolynomials,Prepend[listOfRootNames,Xbaz],CoefficientDomain->RationalFunctions];
,
TEMPgrobBasis=TEMPgrobBasisTry];

positionOfTheLinearPolynomialInXbaz=Position[Exponent[#,Xbaz]&/@TEMPgrobBasis,1][[1,1]];
TEMPsol=Solve[TEMPgrobBasis[[positionOfTheLinearPolynomialInXbaz]]==0,Xbaz];
If[TEMPsol==={},"No solution",(First[TEMPsol][[1,2]]/.RT[ifoo_]:> listOfRoots[[ifoo]])]
];


(* Apply the Gr\[ODoubleDot]bner basis resolution method to each element of an array *)
resolveRootViaGroebnerBasis::nnarg="The size of 'listOfRoots' must match the size of 'listOfRootPowers'! ";



resolveRootViaGroebnerBasisMatrix[arrayToSimplify_,listOfRoots_,listOfRootPowers_]/;If[Length[listOfRoots]==Length[listOfRootPowers],True,Message[resolveRootViaGroebnerBasis::nnarg];False]:=If[listOfRoots==={},
Return[arrayToSimplify],
If[globalSymBuildParallelize,
(*Parallel*)
DistributeDefinitions[resolveRootViaGroebnerBasis,arrayToSimplify,listOfRoots,listOfRootPowers];
PrintTemporary[" Evaluating the Gr\[ODoubleDot]bner basis resolution in parallel. No monitoring is available, be patient...."];
Return[ParallelMap[resolveRootViaGroebnerBasis[#,listOfRoots,listOfRootPowers]&,arrayToSimplify,{2},Method-> "CoarsestGrained"]],
(*Series*)
Return[Monitor[
Table[resolveRootViaGroebnerBasis[arrayToSimplify[[irow,ifoo]],listOfRoots,listOfRootPowers],{irow,1,Dimensions[arrayToSimplify][[1]]},{ifoo,1,Dimensions[arrayToSimplify][[2]]}],
{"row: "<>ToString[irow],"column: "<>ToString[ifoo]}]]
];
];



(*mapMonitor::usage=" A monitoring function for 'Map'.";

mapMonitor[operation_,expr_,lvlSpec_:{1}]:=Block[{Auxf,counter=0},Auxf[x_]:=(counter++;operation[x]);
Monitor[Map[Auxf,expr,lvlSpec],counter]];


parallelMapMonitor::usage=" A rough monitoring function for 'ParallelMap'.";

parallelMapMonitor[operation_,expr_,Threshold_Integer:100,lvlSpec_:{1}]:=Block[{thresh=Threshold,Auxf,counter=0,localcounter},
SetSharedVariable[counter];
ParallelEvaluate[localcounter=1;];
Auxf[x_]:=(If[localcounter\[GreaterEqual]thresh,counter=counter+localcounter;localcounter=1,localcounter++];operation[x]);
Monitor[ParallelMap[Auxf,expr,lvlSpec],{"Rougly at step ", counter}]
];*)


(* ::Section::Initialization:: *)
(*(*Generating the integrability matrix \[DoubleStruckCapitalM] from a list of rational equations*)*)


(* ::Input::Initialization:: *)
buildFMatrixReducedForASetOfEquations[setOfEquations_,allvariables_,sampleSize_,maxSamplePoints_,toleranceForRetries_]:=Module[{
listPrimes,extraSamples,succesfullTry,TEMPfunction,timeMeasure,
TEMPmatrix,TEMPrandomSample, TEMPtryTheFunction,TEMPsetOfEquations,
newVariables,variablesRedef,variablesRedefReverse},

listPrimes=Prime[Range[sampleSize]];

newVariables=Table[ToExpression["xTEMP"<>ToString[i]],{i,1,Length[allvariables]}];
variablesRedef=Table[allvariables[[i]]->newVariables[[i]],{i,1,Length[newVariables]}];
variablesRedefReverse=Table[newVariables[[i]]->allvariables[[i]],{i,1,Length[newVariables]}];
TEMPsetOfEquations=SparseArray[(setOfEquations//ArrayRules)/. variablesRedef];

TEMPmatrix={};

PrintTemporary["Starting to make the irreducible matrix. This might take a while...."];

Monitor[Do[
TEMPfunction[Sequence@@(Pattern[#1,_]&)/@newVariables]:=Evaluate[TEMPsetOfEquations[[fctIter]]];
extraSamples=0;
Do[
timeMeasure=SessionTime[];
(*-------------------------------------------------*)
(* Add a new sample point and avoid singularities *)
succesfullTry=0;
Do[
TEMPrandomSample=RandomSample[listPrimes,Length[newVariables]];
TEMPtryTheFunction=TEMPfunction[Sequence@@TEMPrandomSample]//Quiet;
If[Cases[ArrayRules[TEMPtryTheFunction][[All,2]],ComplexInfinity,Infinity]==={},succesfullTry=1;Break[]]
,
{iterbaz,1,toleranceForRetries}];
(*-------------------------------------------------*)
If[succesfullTry==1,TEMPmatrix=Append[TEMPmatrix,TEMPtryTheFunction];,Return["Error: increase 'toleranceForRetries'"]];
,
{step,1,maxSamplePoints}];
,
{fctIter,1,Length[TEMPsetOfEquations]}],"Adding equation "<>ToString[fctIter]];
Return[rowReduceMatrix[TEMPmatrix//Normal]//SparseArray//sparseArrayZeroRowCut]
];


(* ::Section::Initialization:: *)
(*(*Commands to use when the \[DoubleStruckCapitalM] matrix contains roots*)*)


(* ::Input::Initialization:: *)
takeSecondEntry[array_]:=If[#==={},0,#[[1,2]]]&/@array;


collectRootCoefficients[expressionArray_,namesOfRoots_]:=Module[{arrayOfRootCoefficients=Map[CoefficientRules[#,namesOfRoots]&,expressionArray,{2}], possiblePowers,eqnMatrix},possiblePowers=Sort[DeleteDuplicates[Flatten[Table[arrayOfRootCoefficients[[iter]][[All,All,1]],{iter,1,Length[arrayOfRootCoefficients]}],2]]];
eqnMatrix=Table[takeSecondEntry[Table[Select[arrayOfRootCoefficients[[iterrow,iterbaz]],(#[[1]]===fooPower)&],{fooPower,possiblePowers}]],{iterrow,1,Length[arrayOfRootCoefficients]},{iterbaz,1,Dimensions[arrayOfRootCoefficients][[2]]}];
Flatten[transposeLevelsSparseArray[eqnMatrix//SparseArray,{3,1,2}],1]
];


(* ::Section:: *)
(*Putting all the commands together into one such that it is easy for the user*)


computeTheIntegrabilityTensor::nnarg=" The size of the array 'listOfRoots' must match the size of the array 'listOfRootPowers'!  ";

computeTheIntegrabilityTensor[alphabet_,allvariables_,listOfRoots_,listOfRootPowers_,sampleSize_,maxSamplePoints_,toleranceForRetries_]/;If[Length[listOfRoots]==Length[listOfRootPowers],True,Message[sparseArrayGlue::nnarg];False]:=Module[{TEMPintegrableEquations, TEMPintegrableEquationsResolved,TEMPintegrableEquationsRationalized,TEMPintegrabilityMatrix},
Print["Generating the integrability equations..."];
If[listOfRoots==={},
TEMPintegrableEquationsRationalized=integrableEquationsRational[alphabet,allvariables];
Print["...Done. Generating the integrability tensor..."];
,
TEMPintegrableEquations=integrableEquationsWithRoots[alphabet,allvariables,listOfRoots,listOfRootPowers];
TEMPintegrableEquationsResolved=resolveRootViaGroebnerBasisMatrix[TEMPintegrableEquations,listOfRoots,listOfRootPowers];
TEMPintegrableEquationsRationalized=collectRootCoefficients[TEMPintegrableEquationsResolved,Table[ToExpression["\[Rho]"<>ToString[iter]],{iter,1,Length[listOfRoots]}]];
Print["Done with the resolution of the roots using Gr\[ODoubleDot]bner bases. Generating the integrability tensor..."];
];
TEMPintegrabilityMatrix=buildFMatrixReducedForASetOfEquations[TEMPintegrableEquationsRationalized//Normal//Factor,allvariables,sampleSize,maxSamplePoints,toleranceForRetries];
Return [matrixFReducedToTensor[TEMPintegrabilityMatrix]];
];


(* ::Chapter::Initialization:: *)
(*(*Computing the integrable symbols*)*)


(* ::Section::Initialization:: *)
(*(*Computing the tensors for the integrable symbols*)*)


(* ::Subsubsection::Initialization:: *)
(*(*n-Entry conditions and the weight 1 construction *)*)


(* ::Input::Initialization:: *)



Default[weight1Solution]={};

weight1Solution[alphabet_,forbiddenEntries_.]:=Table[KroneckerDelta[i1,j1],{j0,1,1},{i1,1,Length[alphabet]},{j1,Complement[Range[Length[alphabet]],forbiddenEntries]}]//SparseArray;

Default[weight1SolutionEvenAndOdd]={};

weight1SolutionEvenAndOdd::nnarg=" The dimensions of the matrices are mismatched! ";

weight1SolutionEvenAndOdd[alphabet_,listOfSymbolSigns_,forbiddenEntries_]/;If[Length[alphabet]== Length[listOfSymbolSigns],True,Message[weight1SolutionEvenAndOdd::nnarg];False]:={Table[KroneckerDelta[i1,j1],{j0,1,1},{i1,1,Length[alphabet]},{j1,Complement[Range[Length[alphabet]],forbiddenEntries]}]//SparseArray,Table[listOfSymbolSigns[[j1]],{j1,Complement[Range[Length[alphabet]],forbiddenEntries]}]};


weightLForbiddenSequencesEquationMatrix[allPreviousWeightSymbolsTensorList_,listOfForbiddenSequences_,sizeAlphabet_]:=Module[{tempFullTensor=(dotSymbolTensors[allPreviousWeightSymbolsTensorList])[[1]],dimLastSolutionSpace,preTensor},
dimLastSolutionSpace=Dimensions[Last[allPreviousWeightSymbolsTensorList]]//Last;
(* Make a table, each element of which is Subscript[d, Subscript[j, 1]]^(Subscript[j, 0]Subscript[s^A, 1])....Subscript[d, Subscript[j, L-1]]^(Subscript[j, L-2]Subscript[s^A, L-1]) in a rule form. Then using the rule replacement, multiply it by the tensor Subscript[\[Delta], Subscript[j, L]]^Subscript[S, L]^A. In all of this, s^A={Subscript[(s^A), 1],....Subscript[(s^A), L]} is a forbidden sequence *)
Table[preTensor=(tempFullTensor[[Sequence@@Append[Drop[listOfForbiddenSequences[[forbiddenEntriesElement]],-1],All]]]//ArrayRules);
SparseArray[Drop[preTensor,-1]/.Rule[a__,b_]:>Rule[Append[a,Last[listOfForbiddenSequences[[forbiddenEntriesElement]]]],b],{ dimLastSolutionSpace,sizeAlphabet}]//Flatten,{forbiddenEntriesElement,1,Length[listOfForbiddenSequences]}]//SparseArray];


(* ::Subsubsection::Initialization:: *)
(*(*Computing the next level symbols*)*)


nextWeightSymbolsEquationMatrix[previousWeightSymbolsTensor_,FmatrixTensor_]:=Flatten[transposeLevelsSparseArray[transposeLevelsSparseArray[previousWeightSymbolsTensor,2,3].transposeLevelsSparseArray[FmatrixTensor,1,2],{1,3,2,4}],{{1,2},{3,4}}];




determineNextWeightSymbolsSimple[previousWeightSymbolsTensor_,FmatrixTensor_]:=
Module[{tempEquations,tempSol },
tempEquations=nextWeightSymbolsEquationMatrix[previousWeightSymbolsTensor,FmatrixTensor];
PrintTemporary["Done generating the equations. It's a ", Dimensions[tempEquations], " matrix."];
tempSol=getNullSpace[tempEquations];
solutionSpaceToSymbolsTensor[tempSol,Dimensions[FmatrixTensor][[2]]]];




Default[determineNextWeightSymbols]={};

determineNextWeightSymbols[previousWeightSymbolsTensor_,previousWeightSymbolsSigns_,FmatrixTensor_,listOfSymbolSigns_,forbiddenSequenceConditions_.]:=
Module[{integrabilityEquations,nextWeightNullSpace,sizeAlphabet=Length[listOfSymbolSigns],nextWeightEven,nextWeightOdd},

(*-------------------------------------------*)
(* Get the integrability equations *)

integrabilityEquations=nextWeightSymbolsEquationMatrix[previousWeightSymbolsTensor,FmatrixTensor];
Print["Done computing the integrability equations. Solving...."];

(*-------------------------------------------*)
(* Introduce the weight L entry conditions *)
(* They have to be computed separately using the command 'weightLForbiddenSequencesEquationMatrix' *)
If[!(forbiddenSequenceConditions==={}),integrabilityEquations=sparseArrayGlue[integrabilityEquations,forbiddenSequenceConditions]];

(*-------------------------------------------*)
(* Get the integrable symbols*)
nextWeightNullSpace=getNullSpace[integrabilityEquations];
Print["...done. Separating into even + odd...."];
(*-------------------------------------------*)
(* Compute the even and odd symbols *)
(* If all the symbols are even, bypass the computation *)
If[DeleteDuplicates[Flatten[previousWeightSymbolsSigns]]==={0}&&DeleteDuplicates[Flatten[listOfSymbolSigns]]==={0},
Return[{solutionSpaceToSymbolsTensor[nextWeightNullSpace,sizeAlphabet],Table[0,Length[nextWeightNullSpace]]}]];
(*Otherwise, compute the even and odd conditions and symbols. Make sure that there are no empty arrays....*)
nextWeightEven=getNullSpace[makeTheEvenOddConditionsMatrix[previousWeightSymbolsSigns,listOfSymbolSigns,0].Transpose[nextWeightNullSpace]];
nextWeightOdd=getNullSpace[makeTheEvenOddConditionsMatrix[previousWeightSymbolsSigns,listOfSymbolSigns,1].Transpose[nextWeightNullSpace]];
Print["...done. "];
Which[nextWeightEven==={},
nextWeightOdd=solutionSpaceToSymbolsTensor[nextWeightOdd.nextWeightNullSpace,sizeAlphabet];
Return[{nextWeightOdd,Table[1,Dimensions[nextWeightOdd][[3]]]}];
,
nextWeightOdd==={},
nextWeightEven=solutionSpaceToSymbolsTensor[nextWeightEven.nextWeightNullSpace,sizeAlphabet];
Return[{nextWeightEven,Table[0,Dimensions[nextWeightEven][[3]]]}];
,
True,
nextWeightEven=solutionSpaceToSymbolsTensor[nextWeightEven.nextWeightNullSpace,sizeAlphabet];
nextWeightOdd=solutionSpaceToSymbolsTensor[nextWeightOdd.nextWeightNullSpace,sizeAlphabet];
Return[{integrableSymbolsTensorsGlue[nextWeightEven,nextWeightOdd],Join[Table[0,Dimensions[nextWeightEven][[3]]],Table[1,Dimensions[nextWeightOdd][[3]]]]}];
];
];




(* ::Chapter::Initialization:: *)
(*(*Determine tranformation matrices between alphabets*)*)


buildTransformationMatrix[weightLsymbolTensor_,previousTransformationMatrix_,alphabetTransformationMatrix_,limitAlphabetInversionMatrix_]:=Module[{limitAlphabetSize=Dimensions[alphabetTransformationMatrix][[2]],tempArray},
tempArray=auxFlattenTwoIndices23[transposeLevelsSparseArray[weightLsymbolTensor,{3,1,2}].SparseArray[alphabetTransformationMatrix],limitAlphabetSize].auxFlattenTwoIndices12[SparseArray[previousTransformationMatrix].transposeLevelsSparseArray[inverseMatrixToTensor[limitAlphabetInversionMatrix,limitAlphabetSize],{2,3,1}],limitAlphabetSize];
Return[SparseArray[tempArray,Dimensions[tempArray]]];
];
inverseMatrixToTensor[inverseMatrix_,sizeAlphabet_]:=SparseArray[Most[inverseMatrix//ArrayRules]/. ({a1_,a2_}->a3_):>({a1,Quotient[a2-1,sizeAlphabet]+1,Mod[a2,sizeAlphabet,1]}->a3),{Dimensions[inverseMatrix][[1]],Dimensions[inverseMatrix][[2]]/sizeAlphabet,sizeAlphabet}];



(* ::Chapter:: *)
(*Taking derivatives of symbols*)


(*---------------------------------------------------------------------------*)
(* symbolDerivative with 3 entries, acting on formal sb objects  *)

symbolDerivative[A_,alphabet_,variable_]:=Plus@@(symbolDerivative[#,alphabet,variable]&/@(List@@A))/;Head[A]===Plus&&(!(Head[variable]===List));
symbolDerivative[A_ B_,alphabet_,variable_]:=symbolDerivative[A,alphabet,variable] B+A symbolDerivative[B,alphabet,variable]/;(!(Head[variable]===List));
symbolDerivative[A_,alphabet_,variable_]:=0/;NumericQ[A];
symbolDerivative[sb[\[ScriptCapitalS][X_],A_],alphabet_,variable_]:=If[A==={},derivative[sb[\[ScriptCapitalS][X],{}],variable], sb[\[ScriptCapitalS][X],Drop[A,-1]]D[Log[alphabet[[Last[A]]]],variable]]/;(!(Head[variable]===List));
symbolDerivative[sb[{}],alphabet_,variable_]:=0;
symbolDerivative[sb[A_],alphabet_,variable_]:= sb[Drop[A,-1]]Factor[D[Log[alphabet[[Last[A]]]],variable]]/;(!(Head[variable]===List)); 
symbolDerivative[expression_,alphabet_,variable_]:= D[expression,variable]/;(FreeQ[{expression},sb[A___]]&&FreeQ[{expression},SB[A___]]);
symbolDerivative[derivative[expression_,variable_],alphabet_,variable_]:=derivative[expression,{variable,2}];  
symbolDerivative[derivative[expression_,{variable_,n_}],alphabet_,variable_]:=derivative[expression,{variable,n+1}];

symbolDerivative[expression_,alphabet_,{variable_,n_}]:=Module[{tempExp},
If[n>=0&&Element[n,Integers],
tempExp=expression;
Do[tempExp=symbolDerivative[tempExp,alphabet,variable],{iter,1,n}];
Return[tempExp],
Return["The number of times one differentiates has to be a positive integer!"]
]
];

(*---------------------------------------------------------------------------*)
(* symbolDerivative with 2 entries, acting on formal SB objects *)

symbolDerivative[A_,variable_]:=Plus@@(symbolDerivative[#,variable]&/@(List@@A))/;Head[A]===Plus&&(!(Head[variable]===List));
symbolDerivative[A_ B_,variable_]:=symbolDerivative[A,variable] B+A symbolDerivative[B,variable]/;(!(Head[variable]===List));
symbolDerivative[A_,variable_]:=0/;NumericQ[A];
symbolDerivative[SB[A_],variable_]:= SB[Drop[A,-1]]Factor[D[Log[Last[A]],variable]]/;(!(Head[variable]===List));
symbolDerivative[expression_,variable_]:=D[expression,variable]/;(FreeQ[{expression},sb[A___]]&&FreeQ[{expression},SB[A___]]);

symbolDerivative[expression_,{variable_,n_}]:=Module[{tempExp},
If[n>=0&&Element[n,Integers],
tempExp=expression;
Do[tempExp=symbolDerivative[tempExp,variable],{iter,1,n}];
Return[tempExp],
Return["The number of times one differentiates has to be a positive integer!"]
]
];



(* ::Title::Initialization:: *)
(*(*End *)*)


EndPackage[]
