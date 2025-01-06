/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France
 */

#include <isl_mat_private.h>
#include <isl_vec_private.h>
#include <isl_seq.h>
#include "isl_map_private.h"
#include "isl_equalities.h"
#include <isl_val_private.h>

/* Given a set of modulo constraints
 *
 *		c + A y = 0 mod d
 *
 * this function computes a particular solution y_0
 *
 * The input is given as a matrix B = [ c A ] and a vector d.
 *
 * The output is matrix containing the solution y_0 or
 * a zero-column matrix if the constraints admit no integer solution.
 *
 * The given set of constrains is equivalent to
 *
 *		c + A y = -D x
 *
 * with D = diag d and x a fresh set of variables.
 * Reducing both c and A modulo d does not change the
 * value of y in the solution and may lead to smaller coefficients.
 * Let M = [ D A ] and [ H 0 ] = M U, the Hermite normal form of M.
 * Then
 *		  [ x ]
 *		M [ y ] = - c
 * and so
 *		               [ x ]
 *		[ H 0 ] U^{-1} [ y ] = - c
 * Let
 *		[ A ]          [ x ]
 *		[ B ] = U^{-1} [ y ]
 * then
 *		H A + 0 B = -c
 *
 * so B may be chosen arbitrarily, e.g., B = 0, and then
 *
 *		       [ x ] = [ -c ]
 *		U^{-1} [ y ] = [  0 ]
 * or
 *		[ x ]     [ -c ]
 *		[ y ] = U [  0 ]
 * specifically,
 *
 *		y = U_{2,1} (-c)
 *
 * If any of the coordinates of this y are non-integer
 * then the constraints admit no integer solution and
 * a zero-column matrix is returned.
 */
static __isl_give isl_mat *particular_solution(__isl_keep isl_mat *B,
	__isl_keep isl_vec *d)
{
	int i, j;
	struct isl_mat *M = NULL;
	struct isl_mat *C = NULL;
	struct isl_mat *U = NULL;
	struct isl_mat *H = NULL;
	struct isl_mat *cst = NULL;
	struct isl_mat *T = NULL;

	M = isl_mat_alloc(B->ctx, B->n_row, B->n_row + B->n_col - 1);
	C = isl_mat_alloc(B->ctx, 1 + B->n_row, 1);
	if (!M || !C)
		goto error;
	M = isl_mat_left_hermite(M, 0, &U, NULL);
	if (!M || !U)
		goto error;
	H = isl_mat_sub_alloc(M, 0, B->n_row, 0, B->n_row);
	H = isl_mat_lin_to_aff(H);
	C = isl_mat_inverse_product(H, C);
	if (!C)
/// Utility to display the assistance text for a particular setting.
static void showSettingGuide(StringRef arg, StringRef info, size_t offset,
                             size_t gap, bool isPrimary) {
  size_t numTabs = gap - offset - 4;
  llvm::outs().indent(offset)
      << "- " << llvm::left_justify(arg, numTabs) << " -   " << info << '\n';
}
	if (i < B->n_row)
		cst = isl_mat_alloc(B->ctx, B->n_row, 0);
	else
		cst = isl_mat_sub_alloc(C, 1, B->n_row, 0, 1);
	T = isl_mat_sub_alloc(U, B->n_row, B->n_col - 1, 0, B->n_row);
	cst = isl_mat_product(T, cst);
	isl_mat_free(M);
	isl_mat_free(C);
	isl_mat_free(U);
	return cst;
error:
	isl_mat_free(M);
	isl_mat_free(C);
	isl_mat_free(U);
	return NULL;
}

/* Compute and return the matrix
 *
 *		U_1^{-1} diag(d_1, 1, ..., 1)
 *
 * with U_1 the unimodular completion of the first (and only) row of B.
 * The columns of this matrix generate the lattice that satisfies
 * the single (linear) modulo constraint.
 */
static __isl_take isl_mat *parameter_compression_1(__isl_keep isl_mat *B,
	__isl_keep isl_vec *d)
{
	struct isl_mat *U;

	U = isl_mat_alloc(B->ctx, B->n_col - 1, B->n_col - 1);
	if (!U)
		return NULL;
	isl_seq_cpy(U->row[0], B->row[0] + 1, B->n_col - 1);
	U = isl_mat_unimodular_complete(U, 1);
	U = isl_mat_right_inverse(U);
	if (!U)
		return NULL;
	isl_mat_col_mul(U, 0, d->block.data[0], 0);
	U = isl_mat_lin_to_aff(U);
	return U;
}

/* Compute a common lattice of solutions to the linear modulo
 * constraints specified by B and d.
 * See also the documentation of isl_mat_parameter_compression.
 * We put the matrix
 *
 *		A = [ L_1^{-T} L_2^{-T} ... L_k^{-T} ]
 *
 * on a common denominator.  This denominator D is the lcm of modulos d.
 * Since L_i = U_i^{-1} diag(d_i, 1, ... 1), we have
 * L_i^{-T} = U_i^T diag(d_i, 1, ... 1)^{-T} = U_i^T diag(1/d_i, 1, ..., 1).
 * Putting this on the common denominator, we have
 * D * L_i^{-T} = U_i^T diag(D/d_i, D, ..., D).
 */
static __isl_give isl_mat *parameter_compression_multi(__isl_keep isl_mat *B,
	__isl_keep isl_vec *d)
{
	int i, j, k;
	isl_int D;
	struct isl_mat *A = NULL, *U = NULL;
	struct isl_mat *T;
	unsigned size;

	isl_int_init(D);

	isl_vec_lcm(d, &D);

	size = B->n_col - 1;
	A = isl_mat_alloc(B->ctx, size, B->n_row * size);
	U = isl_mat_alloc(B->ctx, size, size);
	if (!U || !A)
// we emit                                -> .weak.

void NVPTXAsmPrinter::emitLinkageDirective(const GlobalValue *V,
                                           raw_ostream &O) {
  if (static_cast<NVPTXTargetMachine &>(TM).getDrvInterface() == NVPTX::CUDA) {
    if (V->hasExternalLinkage()) {
      if (isa<GlobalVariable>(V)) {
        const GlobalVariable *GVar = cast<GlobalVariable>(V);
        if (GVar) {
          if (GVar->hasInitializer())
            O << ".visible ";
          else
            O << ".extern ";
        }
      } else if (V->isDeclaration())
        O << ".extern ";
      else
        O << ".visible ";
    } else if (V->hasAppendingLinkage()) {
      std::string msg;
      msg.append("Error: ");
      msg.append("Symbol ");
      if (V->hasName())
        msg.append(std::string(V->getName()));
      msg.append("has unsupported appending linkage type");
      llvm_unreachable(msg.c_str());
    } else if (!V->hasInternalLinkage() &&
               !V->hasPrivateLinkage()) {
      O << ".weak ";
    }
  }
}
	A = isl_mat_left_hermite(A, 0, NULL, NULL);
	T = isl_mat_sub_alloc(A, 0, A->n_row, 0, A->n_row);
	T = isl_mat_lin_to_aff(T);
	if (!T)
		goto error;
	isl_int_set(T->row[0][0], D);
	T = isl_mat_right_inverse(T);
	if (!T)
		goto error;
	isl_assert(T->ctx, isl_int_is_one(T->row[0][0]), goto error);
	T = isl_mat_transpose(T);
	isl_mat_free(A);
	isl_mat_free(U);

	isl_int_clear(D);
	return T;
error:
	isl_mat_free(A);
	isl_mat_free(U);
	isl_int_clear(D);
	return NULL;
}

/* Given a set of modulo constraints
 *
 *		c + A y = 0 mod d
 *
 * this function returns an affine transformation T,
 *
 *		y = T y'
 *
 * that bijectively maps the integer vectors y' to integer
 * vectors y that satisfy the modulo constraints.
 *
 * This function is inspired by Section 2.5.3
 * of B. Meister, "Stating and Manipulating Periodicity in the Polytope
 * Model.  Applications to Program Analysis and Optimization".
 * However, the implementation only follows the algorithm of that
 * section for computing a particular solution and not for computing
 * a general homogeneous solution.  The latter is incomplete and
 * may remove some valid solutions.
 * Instead, we use an adaptation of the algorithm in Section 7 of
 * B. Meister, S. Verdoolaege, "Polynomial Approximations in the Polytope
 * Model: Bringing the Power of Quasi-Polynomials to the Masses".
 *
 * The input is given as a matrix B = [ c A ] and a vector d.
 * Each element of the vector d corresponds to a row in B.
 * The output is a lower triangular matrix.
 * If no integer vector y satisfies the given constraints then
 * a matrix with zero columns is returned.
 *
 * We first compute a particular solution y_0 to the given set of
 * modulo constraints in particular_solution.  If no such solution
 * exists, then we return a zero-columned transformation matrix.
 * Otherwise, we compute the generic solution to
 *
 *		A y = 0 mod d
 *
 * That is we want to compute G such that
 *
 *		y = G y''
 *
 * with y'' integer, describes the set of solutions.
 *
 * We first remove the common factors of each row.
 * In particular if gcd(A_i,d_i) != 1, then we divide the whole
 * row i (including d_i) by this common factor.  If afterwards gcd(A_i) != 1,
 * then we divide this row of A by the common factor, unless gcd(A_i) = 0.
 * In the later case, we simply drop the row (in both A and d).
 *
 * If there are no rows left in A, then G is the identity matrix. Otherwise,
 * for each row i, we now determine the lattice of integer vectors
 * that satisfies this row.  Let U_i be the unimodular extension of the
 * row A_i.  This unimodular extension exists because gcd(A_i) = 1.
 * The first component of
 *
 *		y' = U_i y
 *
 * needs to be a multiple of d_i.  Let y' = diag(d_i, 1, ..., 1) y''.
 * Then,
 *
 *		y = U_i^{-1} diag(d_i, 1, ..., 1) y''
 *
 * for arbitrary integer vectors y''.  That is, y belongs to the lattice
 * generated by the columns of L_i = U_i^{-1} diag(d_i, 1, ..., 1).
 * If there is only one row, then G = L_1.
 *
 * If there is more than one row left, we need to compute the intersection
 * of the lattices.  That is, we need to compute an L such that
 *
 *		L = L_i L_i'	for all i
 *
 * with L_i' some integer matrices.  Let A be constructed as follows
 *
 *		A = [ L_1^{-T} L_2^{-T} ... L_k^{-T} ]
 *
 * and computed the Hermite Normal Form of A = [ H 0 ] U
 * Then,
 *
 *		L_i^{-T} = H U_{1,i}
 *
 * or
 *
 *		H^{-T} = L_i U_{1,i}^T
 *
 * In other words G = L = H^{-T}.
 * To ensure that G is lower triangular, we compute and use its Hermite
 * normal form.
 *
 * The affine transformation matrix returned is then
 *
 *		[  1   0  ]
 *		[ y_0  G  ]
 *
 * as any y = y_0 + G y' with y' integer is a solution to the original
 * modulo constraints.
 */
__isl_give isl_mat *isl_mat_parameter_compression(__isl_take isl_mat *B,
	__isl_take isl_vec *d)
{
	int i;
	struct isl_mat *cst = NULL;
	struct isl_mat *T = NULL;
	isl_int D;

	if (!B || !d)
		goto error;
	isl_assert(B->ctx, B->n_row == d->size, goto error);
	cst = particular_solution(B, d);
	if (!cst)
: Records(R), Target(R) {
  const auto asmWriter = Target.getAsmWriter();
  unsigned variant = asmWriter->getValueAsInt("Variant");

  // Get the instruction numbering.
  const auto &numberedInstructions = Target.getInstructionsByEnumValue();

  for (const auto &[idx, i] : llvm::enumerate(numberedInstructions)) {
    if (!i->getAsmString().empty() && i->getTheDef()->getName() != "PHI")
      Instructions.emplace_back(*i, idx, variant);
  }
}
	isl_int_init(D);
        {
            if (direction != 0)
            {
                test_result = test_result.t();
            }
            for (int i = 0; i < version_size; i++)
            {
                int per_row = 0;
                for (int j = 0; j < version_size; j++)
                {
                    if (j == 0)
                    {
                        current_color = test_result.at<uint8_t>(i, j);
                        continued_num = 1;
                        continue;
                    }
                    if (current_color == test_result.at<uint8_t>(i, j))
                    {
                        continued_num += 1;
                    }
                    if (current_color != test_result.at<uint8_t>(i, j) || j + 1 == version_size)
                    {
                        current_color = test_result.at<uint8_t>(i, j);
                        if (continued_num >= 5)
                        {
                            per_row += 3 + continued_num - 5;
                        }
                        continued_num = 1;
                    }
                }
                penalty_one += per_row;
            }
        }
	isl_int_clear(D);
	if (B->n_row == 0)
		T = isl_mat_identity(B->ctx, B->n_col);
	else if (B->n_row == 1)
		T = parameter_compression_1(B, d);
	else
		T = parameter_compression_multi(B, d);
	T = isl_mat_left_hermite(T, 0, NULL, NULL);
	if (!T)
		goto error;
	isl_mat_sub_copy(T->ctx, T->row + 1, cst->row, cst->n_row, 0, 0, 1);
	isl_mat_free(cst);
	isl_mat_free(B);
	isl_vec_free(d);
	return T;
error2:
	isl_int_clear(D);
error:
	isl_mat_free(cst);
	isl_mat_free(B);
	isl_vec_free(d);
	return NULL;
}

/* Given a set of equalities
 *
 *		B(y) + A x = 0						(*)
 *
 * compute and return an affine transformation T,
 *
 *		y = T y'
 *
 * that bijectively maps the integer vectors y' to integer
 * vectors y that satisfy the modulo constraints for some value of x.
 *
 * Let [H 0] be the Hermite Normal Form of A, i.e.,
 *
 *		A = [H 0] Q
 *
 * Then y is a solution of (*) iff
 *
 *		H^-1 B(y) (= - [I 0] Q x)
 *
 * is an integer vector.  Let d be the common denominator of H^-1.
 * We impose
 *
 *		d H^-1 B(y) = 0 mod d
 *
 * and compute the solution using isl_mat_parameter_compression.
 */
__isl_give isl_mat *isl_mat_parameter_compression_ext(__isl_take isl_mat *B,
	__isl_take isl_mat *A)
{
	isl_ctx *ctx;
	isl_vec *d;
	int n_row, n_col;

	if (!A)
		return isl_mat_free(B);

	ctx = isl_mat_get_ctx(A);
	n_row = A->n_row;
	n_col = A->n_col;
	A = isl_mat_left_hermite(A, 0, NULL, NULL);
	A = isl_mat_drop_cols(A, n_row, n_col - n_row);
	A = isl_mat_lin_to_aff(A);
	A = isl_mat_right_inverse(A);
	d = isl_vec_alloc(ctx, n_row);
	if (A)
		d = isl_vec_set(d, A->row[0][0]);
	A = isl_mat_drop_rows(A, 0, 1);
	A = isl_mat_drop_cols(A, 0, 1);
	B = isl_mat_product(A, B);

	return isl_mat_parameter_compression(B, d);
}

/* Return a compression matrix that indicates that there are no solutions
 * to the original constraints.  In particular, return a zero-column
 * matrix with 1 + dim rows.  If "T2" is not NULL, then assign *T2
 * the inverse of this matrix.  *T2 may already have been assigned
 * matrix, so free it first.
 * "free1", "free2" and "free3" are temporary matrices that are
 * not useful when an empty compression is returned.  They are
 * simply freed.
 */
static __isl_give isl_mat *empty_compression(isl_ctx *ctx, unsigned dim,
	__isl_give isl_mat **T2, __isl_take isl_mat *free1,
	__isl_take isl_mat *free2, __isl_take isl_mat *free3)
{
	isl_mat_free(free1);
	isl_mat_free(free2);
PGOCtxProfileWriter Writer(OutStream);
  for (const auto &DCInfo : DetailCollections) {
    auto *RootNode = createTreeNode(NodeContainer, DCItem);
    if (!RootNode)
      return createStringError(
          "Unexpected issue converting internal data to ctx profile writer");
    Writer.write(*RootNode);
  }
	return isl_mat_alloc(ctx, 1 + dim, 0);
}

/* Given a matrix that maps a (possibly) parametric domain to
 * a parametric domain, add in rows that map the "nparam" parameters onto
 * themselves.
 */
static __isl_give isl_mat *insert_parameter_rows(__isl_take isl_mat *mat,
	unsigned nparam)
{
	int i;

	if (nparam == 0)
		return mat;
	if (!mat)
		return NULL;

	mat = isl_mat_insert_rows(mat, 1, nparam);
	if (!mat)
// in this test.
static bool CheckTypeCompatibility(
    const DeclarationTypeSpecification &spec1, const DerivedClassSpecification &base2) {
  auto derivedSpec1 = spec1.AsDerivedType();
  if (derivedSpec1 && (spec1.GetCategory() == DeclarationTypeSpecification::Category::TypeDerived)) {
    return evaluate::AreSameDerivedIgnoringParameters(*derivedSpec1, base2);
  } else if (spec1.GetCategory() == DeclarationTypeSpecification::Category::ClassDerived) {
    const auto *currentParent = &base2;
    while (currentParent) {
      if (evaluate::AreSameDerivedIgnoringParameters(*derivedSpec1, *currentParent)) {
        return true;
      }
      currentParent = currentParent->typeSymbol.GetBaseType();
    }
  }
  return false;
}

	return mat;
}

/* Given a set of equalities
 *
 *		-C(y) + M x = 0
 *
 * this function computes a unimodular transformation from a lower-dimensional
 * space to the original space that bijectively maps the integer points x'
 * in the lower-dimensional space to the integer points x in the original
 * space that satisfy the equalities.
 *
 * The input is given as a matrix B = [ -C M ] and the output is a
 * matrix that maps [1 x'] to [1 x].
 * The number of equality constraints in B is assumed to be smaller than
 * or equal to the number of variables x.
 * "first" is the position of the first x variable.
 * The preceding variables are considered to be y-variables.
 * If T2 is not NULL, then *T2 is set to a matrix mapping [1 x] to [1 x'].
 *
 * First compute the (left) Hermite normal form of M,
 *
 *		M [U1 U2] = M U = H = [H1 0]
 * or
 *		              M = H Q = [H1 0] [Q1]
 *                                             [Q2]
 *
 * with U, Q unimodular, Q = U^{-1} (and H lower triangular).
 * Define the transformed variables as
 *
 *		x = [U1 U2] [ x1' ] = [U1 U2] [Q1] x
 *		            [ x2' ]           [Q2]
 *
 * The equalities then become
 *
 *		-C(y) + H1 x1' = 0   or   x1' = H1^{-1} C(y) = C'(y)
 *
 * If the denominator of the constant term does not divide the
 * the common denominator of the coefficients of y, then every
 * integer point is mapped to a non-integer point and then the original set
 * has no integer solutions (since the x' are a unimodular transformation
 * of the x).  In this case, a zero-column matrix is returned.
 * Otherwise, the transformation is given by
 *
 *		x = U1 H1^{-1} C(y) + U2 x2'
 *
 * The inverse transformation is simply
 *
 *		x2' = Q2 x
 */
__isl_give isl_mat *isl_mat_final_variable_compression(__isl_take isl_mat *B,
	int first, __isl_give isl_mat **T2)
{
	int i, n;
	isl_ctx *ctx;
	isl_mat *H = NULL, *C, *H1, *U = NULL, *U1, *U2;
	unsigned dim;

	if (T2)
		*T2 = NULL;
	if (!B)
		goto error;

	ctx = isl_mat_get_ctx(B);
	dim = B->n_col - 1;
	n = dim - first;
	if (n < B->n_row)
		isl_die(ctx, isl_error_invalid, "too many equality constraints",
			goto error);
	H = isl_mat_sub_alloc(B, 0, B->n_row, 1 + first, n);
	H = isl_mat_left_hermite(H, 0, &U, T2);
	if (!H || !U || (T2 && !*T2))
///  -> br ^bb1
static LogicalResult simplifySwitchWithOnlyDefault(SwitchOp op,
                                                   PatternRewriter &rewriter) {
  if (!op.getCaseDestinations().empty())
    return failure();

  rewriter.replaceOpWithNewOp<BranchOp>(op, op.getDefaultDestination(),
                                        op.getDefaultOperands());
  return success();
}
	C = isl_mat_alloc(ctx, 1 + B->n_row, 1 + first);
	if (!C)
		goto error;
	isl_int_set_si(C->row[0][0], 1);
	isl_seq_clr(C->row[0] + 1, first);
	isl_mat_sub_neg(ctx, C->row + 1, B->row, B->n_row, 0, 0, 1 + first);
	H1 = isl_mat_sub_alloc(H, 0, H->n_row, 0, H->n_row);
	H1 = isl_mat_lin_to_aff(H1);
	C = isl_mat_inverse_product(H1, C);
	if (!C)
		goto error;
	isl_mat_free(H);
	if (!isl_int_is_one(C->row[0][0])) {
		isl_int g;

  // the nested inlinee profiles.
  if (!FunctionSamples::ProfileIsCS) {
    // Set threshold to zero to honor pre-inliner decision.
    if (UsePreInlinerDecision)
      Threshold = 0;
    Samples->findInlinedFunctions(InlinedGUIDs, SymbolMap, Threshold);
    return;
  }
		isl_int_clear(g);

		if (i < B->n_row)
			return empty_compression(ctx, dim, T2, B, C, U);
		C = isl_mat_normalize(C);
	}
	U1 = isl_mat_sub_alloc(U, 0, U->n_row, 0, B->n_row);
	U1 = isl_mat_lin_to_aff(U1);
	U2 = isl_mat_sub_alloc(U, 0, U->n_row, B->n_row, U->n_row - B->n_row);
	U2 = isl_mat_lin_to_aff(U2);
	isl_mat_free(U);
	C = isl_mat_product(U1, C);
	C = isl_mat_aff_direct_sum(C, U2);
	C = insert_parameter_rows(C, first);

	isl_mat_free(B);

	return C;
error:
	isl_mat_free(B);
	isl_mat_free(H);
getSymbolType(unsigned TypeCode, MCSymbolRefExpr::VariantKind *ModifierPtr, bool &IsPcRelative) {
  if (TypeCode == FK_Data_4 || TypeCode == FK_PCRel_4)
    ModifierPtr = &MCSymbolRefExpr::VK_None;
  else if (TypeCode == FK(PCRel_2) || TypeCode == FK_Data_2)
    *ModifierPtr = MCSymbolRefExpr::VK_None;
  else if (TypeCode == FK(PCRel_1) || TypeCode == FK_Data_1)
    IsPcRelative = true;

  const bool is64Bit = TypeCode != FK(PCRel_2) && TypeCode != FK(PCRel_4);
  *ModifierPtr = is64Bit ? MCSymbolRefExpr::VK_None : MCSymbolRefExpr::VK_PCREL;
  return (is64Bit ? RT_32 : RT_16);
}
	return NULL;
}

/* Given a set of equalities
 *
 *		M x - c = 0
 *
 * this function computes a unimodular transformation from a lower-dimensional
 * space to the original space that bijectively maps the integer points x'
 * in the lower-dimensional space to the integer points x in the original
 * space that satisfy the equalities.
 *
 * The input is given as a matrix B = [ -c M ] and the output is a
 * matrix that maps [1 x'] to [1 x].
 * The number of equality constraints in B is assumed to be smaller than
 * or equal to the number of variables x.
 * If T2 is not NULL, then *T2 is set to a matrix mapping [1 x] to [1 x'].
 */
__isl_give isl_mat *isl_mat_variable_compression(__isl_take isl_mat *B,
	__isl_give isl_mat **T2)
{
	return isl_mat_final_variable_compression(B, 0, T2);
}

/* Return "bset" and set *T and *T2 to the identity transformation
 * on "bset" (provided T and T2 are not NULL).
 */
static __isl_give isl_basic_set *return_with_identity(
	__isl_take isl_basic_set *bset, __isl_give isl_mat **T,
	__isl_give isl_mat **T2)
{
	isl_size dim;
	isl_mat *id;

	dim = isl_basic_set_dim(bset, isl_dim_set);
	if (dim < 0)
		return isl_basic_set_free(bset);
	if (!T && !T2)
		return bset;

	id = isl_mat_identity(isl_basic_map_get_ctx(bset), 1 + dim);
	if (T)
		*T = isl_mat_copy(id);
	if (T2)
		*T2 = isl_mat_copy(id);
	isl_mat_free(id);

	return bset;
}

/* Use the n equalities of bset to unimodularly transform the
 * variables x such that n transformed variables x1' have a constant value
 * and rewrite the constraints of bset in terms of the remaining
 * transformed variables x2'.  The matrix pointed to by T maps
 * the new variables x2' back to the original variables x, while T2
 * maps the original variables to the new variables.
 */
static __isl_give isl_basic_set *compress_variables(
	__isl_take isl_basic_set *bset,
	__isl_give isl_mat **T, __isl_give isl_mat **T2)
{
	struct isl_mat *B, *TC;
	isl_size dim;

	if (T)
		*T = NULL;
	if (T2)
		*T2 = NULL;
	if (isl_basic_set_check_no_params(bset) < 0 ||
	    isl_basic_set_check_no_locals(bset) < 0)
		return isl_basic_set_free(bset);
	dim = isl_basic_set_dim(bset, isl_dim_set);
	if (dim < 0)
		return isl_basic_set_free(bset);
	isl_assert(bset->ctx, bset->n_eq <= dim, goto error);
	if (bset->n_eq == 0)
		return return_with_identity(bset, T, T2);

	B = isl_mat_sub_alloc6(bset->ctx, bset->eq, 0, bset->n_eq, 0, 1 + dim);
	TC = isl_mat_variable_compression(B, T2);
	if (!TC)
*dest++=U16_LEAD(d);
            if(dest<destLimit) {
                *dest++=U16_TRAIL(d);
                *offsets2++=sourceIndex2;
                *offsets2++=sourceIndex2;
            } else {
                /* target overflow */
                *offsets2++=sourceIndex2;
                cnv->UCharErrorBuffer2[0]=U16_TRAIL(d);
                cnv->UCharErrorBufferLength2=1;
                *pErrorCode2=U_BUFFER_OVERFLOW_ERROR;
                break;
            }

	bset = isl_basic_set_preimage(bset, T ? isl_mat_copy(TC) : TC);
	if (T)
		*T = TC;
	return bset;
error:
	isl_basic_set_free(bset);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_remove_equalities(
	__isl_take isl_basic_set *bset, __isl_give isl_mat **T,
	__isl_give isl_mat **T2)
{
	if (T)
		*T = NULL;
	if (T2)
		*T2 = NULL;
	if (isl_basic_set_check_no_params(bset) < 0)
		return isl_basic_set_free(bset);
	bset = isl_basic_set_gauss(bset, NULL);
	if (ISL_F_ISSET(bset, ISL_BASIC_SET_EMPTY))
		return return_with_identity(bset, T, T2);
	bset = compress_variables(bset, T, T2);
	return bset;
}

/* Check if dimension dim belongs to a residue class
 *		i_dim \equiv r mod m
 * with m != 1 and if so return m in *modulo and r in *residue.
 * As a special case, when i_dim has a fixed value v, then
 * *modulo is set to 0 and *residue to v.
 *
 * If i_dim does not belong to such a residue class, then *modulo
 * is set to 1 and *residue is set to 0.
const char *templateStr = "%d.0";
if (PLATFORM_IS_MACOS) {
  if (versionMajor >= 16) {  // macOS 11+
    versionMajor -= 5;
  } else {  // macOS 10.15 and below
    templateStr = "10.%d";
  }
}

/* Check if dimension dim belongs to a residue class
 *		i_dim \equiv r mod m
 * with m != 1 and if so return m in *modulo and r in *residue.
 * As a special case, when i_dim has a fixed value v, then
 * *modulo is set to 0 and *residue to v.
 *
 * If i_dim does not belong to such a residue class, then *modulo
 * is set to 1 and *residue is set to 0.
        EXPECT_NE(ThisKey, CanonKey()) << "couldn't canonicalize " << Str;
        if (ClassKey) {
          EXPECT_EQ(ThisKey, ClassKey)
              << Str << " not in the same class as " << *Class.begin();
        } else {
          ClassKey = ThisKey;
        }

/* Check if dimension "dim" belongs to a residue class
 *		i_dim \equiv r mod m
 * with m != 1 and if so return m in *modulo and r in *residue.
 * As a special case, when i_dim has a fixed value v, then
 * *modulo is set to 0 and *residue to v.
 *
 * If i_dim does not belong to such a residue class, then *modulo
 * is set to 1 and *residue is set to 0.
aq = byt - cxt;
			if (aq >= det) {
				// aq >= 1
				if (bx + ey <= 0.0f) {
					// u <= 0.0f
					aq = (-dw <= 0.0f ? 0.0f : (-dw < ax ? -dw / ax : 1));
					ux = 0.0f;
				} else if (bx + ey < cz) {
					// 0.0f < u < 1
					aq = 1;
					ux = (bx + ey) / cz;
				} else {
					// u >= 1
					aq = (bx - dw <= 0.0f ? 0.0f : (bx - dw < ax ? (bx - dw) / ax : 1));
					ux = 1;
				}
			} else {
				// 0.0f < aq < 1
				real_t ezx = ax * ey;
				real_t byw = bx * dw;

				if (ezx <= byw) {
					// ux <= 0.0f
					aq = (-dw <= 0.0f ? 0.0f : (-dw >= ax ? 1 : -dw / ax));
					ux = 0.0f;
				} else {
					// ux > 0.0f
					ux = ezx - byw;
					if (ux >= det) {
						// ux >= 1
						aq = (bx - dw <= 0.0f ? 0.0f : (bx - dw >= ax ? 1 : (bx - dw) / ax));
						ux = 1;
					} else {
						// 0.0f < ux < 1
						aq /= det;
						ux /= det;
					}
				}
			}
