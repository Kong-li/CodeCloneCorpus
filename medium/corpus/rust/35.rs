//! Inference of closure parameter types based on the closure's expected type.

use std::{cmp, convert::Infallible, mem};

use chalk_ir::{
    cast::Cast,
    fold::{FallibleTypeFolder, TypeFoldable},
    BoundVar, DebruijnIndex, FnSubst, Mutability, TyKind,
};
use either::Either;
use hir_def::{
    data::adt::VariantData,
    hir::{
        Array, AsmOperand, BinaryOp, BindingId, CaptureBy, Expr, ExprId, ExprOrPatId, Pat, PatId,
        Statement, UnaryOp,
    },
    lang_item::LangItem,
    path::Path,
    resolver::ValueNs,
    DefWithBodyId, FieldId, HasModule, TupleFieldId, TupleId, VariantId,
};
use hir_expand::name::Name;
use intern::sym;
use rustc_hash::FxHashMap;
use smallvec::{smallvec, SmallVec};
use stdx::{format_to, never};
use syntax::utils::is_raw_identifier;

use crate::{
    db::{HirDatabase, InternedClosure},
    error_lifetime, from_chalk_trait_id, from_placeholder_idx,
    generics::Generics,
    infer::coerce::CoerceNever,
    make_binders,
    mir::{BorrowKind, MirSpan, MutBorrowKind, ProjectionElem},
    to_chalk_trait_id,
    traits::FnTrait,
    utils::{self, elaborate_clause_supertraits},
    Adjust, Adjustment, AliasEq, AliasTy, Binders, BindingMode, ChalkTraitId, ClosureId, DynTy,
    DynTyExt, FnAbi, FnPointer, FnSig, Interner, OpaqueTy, ProjectionTyExt, Substitution, Ty,
    TyExt, WhereClause,
};

use super::{Expectation, InferenceContext};

impl InferenceContext<'_> {
    // This function handles both closures and coroutines.
    pub(super) fn deduce_closure_type_from_expectations(
        &mut self,
        closure_expr: ExprId,
        closure_ty: &Ty,
        sig_ty: &Ty,
        expectation: &Expectation,
    ) {
        let expected_ty = match expectation.to_option(&mut self.table) {
            Some(ty) => ty,
            None => return,
        };

        if let TyKind::Closure(closure_id, _) = closure_ty.kind(Interner) {
            if let Some(closure_kind) = self.deduce_closure_kind_from_expectations(&expected_ty) {
                self.result
                    .closure_info
                    .entry(*closure_id)
                    .or_insert_with(|| (Vec::new(), closure_kind));
            }
        }

        // Deduction from where-clauses in scope, as well as fn-pointer coercion are handled here.
        let _ = self.coerce(Some(closure_expr), closure_ty, &expected_ty, CoerceNever::Yes);

        // Coroutines are not Fn* so return early.
        if matches!(closure_ty.kind(Interner), TyKind::Coroutine(..)) {
            return;
        }

        // Deduction based on the expected `dyn Fn` is done separately.
        if let TyKind::Dyn(dyn_ty) = expected_ty.kind(Interner) {
            if let Some(sig) = self.deduce_sig_from_dyn_ty(dyn_ty) {
                let expected_sig_ty = TyKind::Function(sig).intern(Interner);

                self.unify(sig_ty, &expected_sig_ty);
            }
        }
    }

    // Closure kind deductions are mostly from `rustc_hir_typeck/src/closure.rs`.
    // Might need to port closure sig deductions too.
    fn deduce_closure_kind_from_expectations(&mut self, expected_ty: &Ty) -> Option<FnTrait> {
        match expected_ty.kind(Interner) {
            TyKind::Alias(AliasTy::Opaque(OpaqueTy { .. })) | TyKind::OpaqueType(..) => {
                let clauses = expected_ty
                    .impl_trait_bounds(self.db)
                    .into_iter()
                    .flatten()
                    .map(|b| b.into_value_and_skipped_binders().0);
                self.deduce_closure_kind_from_predicate_clauses(clauses)
            }
            TyKind::Dyn(dyn_ty) => dyn_ty.principal_id().and_then(|trait_id| {
                self.fn_trait_kind_from_trait_id(from_chalk_trait_id(trait_id))
            }),
            TyKind::InferenceVar(ty, chalk_ir::TyVariableKind::General) => {
                let clauses = self.clauses_for_self_ty(*ty);
                self.deduce_closure_kind_from_predicate_clauses(clauses.into_iter())
            }
            TyKind::Function(_) => Some(FnTrait::Fn),
            _ => None,
        }
    }

    fn deduce_closure_kind_from_predicate_clauses(
        &self,
        clauses: impl DoubleEndedIterator<Item = WhereClause>,
    ) -> Option<FnTrait> {
        let mut expected_kind = None;

        for clause in elaborate_clause_supertraits(self.db, clauses.rev()) {
            let trait_id = match clause {
                WhereClause::AliasEq(AliasEq {
                    alias: AliasTy::Projection(projection), ..
                }) => Some(projection.trait_(self.db)),
                WhereClause::Implemented(trait_ref) => {
                    Some(from_chalk_trait_id(trait_ref.trait_id))
                }
                _ => None,
            };
            if let Some(closure_kind) =
                trait_id.and_then(|trait_id| self.fn_trait_kind_from_trait_id(trait_id))
            {
                // `FnX`'s variants order is opposite from rustc, so use `cmp::max` instead of `cmp::min`
                expected_kind = Some(
                    expected_kind
                        .map_or_else(|| closure_kind, |current| cmp::max(current, closure_kind)),
                );
            }
        }

        expected_kind
    }

    fn deduce_sig_from_dyn_ty(&self, dyn_ty: &DynTy) -> Option<FnPointer> {
        // Search for a predicate like `<$self as FnX<Args>>::Output == Ret`

        let fn_traits: SmallVec<[ChalkTraitId; 3]> =
            utils::fn_traits(self.db.upcast(), self.owner.module(self.db.upcast()).krate())
                .map(to_chalk_trait_id)
                .collect();

        let self_ty = self.result.standard_types.unknown.clone();
        let bounds = dyn_ty.bounds.clone().substitute(Interner, &[self_ty.cast(Interner)]);
        for bound in bounds.iter(Interner) {
            // NOTE(skip_binders): the extracted types are rebound by the returned `FnPointer`
            if let WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(projection), ty }) =
                bound.skip_binders()
            {
                let assoc_data = self.db.associated_ty_data(projection.associated_ty_id);
                if !fn_traits.contains(&assoc_data.trait_id) {
                    return None;
                }

                // Skip `Self`, get the type argument.
                let arg = projection.substitution.as_slice(Interner).get(1)?;
                if let Some(subst) = arg.ty(Interner)?.as_tuple() {
                    let generic_args = subst.as_slice(Interner);
                    let mut sig_tys = Vec::with_capacity(generic_args.len() + 1);
                    for arg in generic_args {
                        sig_tys.push(arg.ty(Interner)?.clone());
                    }
                    sig_tys.push(ty.clone());

                    cov_mark::hit!(dyn_fn_param_informs_call_site_closure_signature);
                    return Some(FnPointer {
                        num_binders: bound.len(Interner),
                        sig: FnSig {
                            abi: FnAbi::RustCall,
                            safety: chalk_ir::Safety::Safe,
                            variadic: false,
                        },
                        substitution: FnSubst(Substitution::from_iter(Interner, sig_tys)),
                    });
                }
            }
        }

        None
    }

    fn fn_trait_kind_from_trait_id(&self, trait_id: hir_def::TraitId) -> Option<FnTrait> {
        FnTrait::from_lang_item(self.db.lang_attr(trait_id.into())?)
    }
}

// The below functions handle capture and closure kind (Fn, FnMut, ..)

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct HirPlace {
    pub(crate) local: BindingId,
    pub(crate) projections: Vec<ProjectionElem<Infallible, Ty>>,
}

impl HirPlace {
    fn ty(&self, ctx: &mut InferenceContext<'_>) -> Ty {
        let mut ty = ctx.table.resolve_completely(ctx.result[self.local].clone());
        for p in &self.projections {
            ty = p.projected_ty(
                ty,
                ctx.db,
                |_, _, _| {
                    unreachable!("Closure field only happens in MIR");
                },
                ctx.owner.module(ctx.db.upcast()).krate(),
            );
        }
        ty
    }

    fn capture_kind_of_truncated_place(
        &self,
        mut current_capture: CaptureKind,
        len: usize,
    ) -> CaptureKind {
        if let CaptureKind::ByRef(BorrowKind::Mut {
            kind: MutBorrowKind::Default | MutBorrowKind::TwoPhasedBorrow,
        }) = current_capture
        {
            if self.projections[len..].iter().any(|it| *it == ProjectionElem::Deref) {
                current_capture =
                    CaptureKind::ByRef(BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture });
            }
        }
        current_capture
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CaptureKind {
    ByRef(BorrowKind),
    ByValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedItem {
    pub(crate) place: HirPlace,
    pub(crate) kind: CaptureKind,
    /// The inner vec is the stacks; the outer vec is for each capture reference.
    ///
    /// Even though we always report only the last span (i.e. the most inclusive span),
    /// we need to keep them all, since when a closure occurs inside a closure, we
    /// copy all captures of the inner closure to the outer closure, and then we may
    /// truncate them, and we want the correct span to be reported.
    span_stacks: SmallVec<[SmallVec<[MirSpan; 3]>; 3]>,
    pub(crate) ty: Binders<Ty>,
}

impl CapturedItem {
    pub fn local(&self) -> BindingId {
        self.place.local
    }

    /// Returns whether this place has any field (aka. non-deref) projections.
    pub fn has_field_projections(&self) -> bool {
        self.place.projections.iter().any(|it| !matches!(it, ProjectionElem::Deref))
    }

    pub fn ty(&self, subst: &Substitution) -> Ty {
        self.ty.clone().substitute(Interner, utils::ClosureSubst(subst).parent_subst())
    }

    pub fn kind(&self) -> CaptureKind {
        self.kind
    }

    pub fn spans(&self) -> SmallVec<[MirSpan; 3]> {
        self.span_stacks.iter().map(|stack| *stack.last().expect("empty span stack")).collect()
    }

    /// Converts the place to a name that can be inserted into source code.
    pub fn place_to_name(&self, owner: DefWithBodyId, db: &dyn HirDatabase) -> String {
        let body = db.body(owner);
        let mut result = body[self.place.local].name.unescaped().display(db.upcast()).to_string();
        for proj in &self.place.projections {
            match proj {
                ProjectionElem::Deref => {}
                ProjectionElem::Field(Either::Left(f)) => {
                    match &*f.parent.variant_data(db.upcast()) {
                        VariantData::Record { fields, .. } => {
                            result.push('_');
                            result.push_str(fields[f.local_id].name.as_str())
                        }
                        VariantData::Tuple { fields, .. } => {
                            let index = fields.iter().position(|it| it.0 == f.local_id);
                            if let Some(index) = index {
                                format_to!(result, "_{index}");
                            }
                        }
                        VariantData::Unit => {}
                    }
                }
                ProjectionElem::Field(Either::Right(f)) => format_to!(result, "_{}", f.index),
                &ProjectionElem::ClosureField(field) => format_to!(result, "_{field}"),
                ProjectionElem::Index(_)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::OpaqueCast(_) => {
                    never!("Not happen in closure capture");
                    continue;
                }
            }
        }
        if is_raw_identifier(&result, db.crate_graph()[owner.module(db.upcast()).krate()].edition) {
            result.insert_str(0, "r#");
        }
        result
    }

    pub fn display_place_source_code(&self, owner: DefWithBodyId, db: &dyn HirDatabase) -> String {
        let body = db.body(owner);
        let krate = owner.krate(db.upcast());
        let edition = db.crate_graph()[krate].edition;
        let mut result = body[self.place.local].name.display(db.upcast(), edition).to_string();
        for proj in &self.place.projections {
            match proj {
                // In source code autoderef kicks in.
                ProjectionElem::Deref => {}
                ProjectionElem::Field(Either::Left(f)) => {
                    let variant_data = f.parent.variant_data(db.upcast());
                    match &*variant_data {
                        VariantData::Record { fields, .. } => format_to!(
                            result,
                            ".{}",
                            fields[f.local_id].name.display(db.upcast(), edition)
                        ),
                        VariantData::Tuple { fields, .. } => format_to!(
                            result,
                            ".{}",
                            fields.iter().position(|it| it.0 == f.local_id).unwrap_or_default()
                        ),
                        VariantData::Unit => {}
                    }
                }
                ProjectionElem::Field(Either::Right(f)) => {
                    let field = f.index;
                    format_to!(result, ".{field}");
                }
                &ProjectionElem::ClosureField(field) => {
                    format_to!(result, ".{field}");
                }
                ProjectionElem::Index(_)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::OpaqueCast(_) => {
                    never!("Not happen in closure capture");
                    continue;
                }
            }
        }
        let final_derefs_count = self
            .place
            .projections
            .iter()
            .rev()
            .take_while(|proj| matches!(proj, ProjectionElem::Deref))
            .count();
        result.insert_str(0, &"*".repeat(final_derefs_count));
        result
    }

    pub fn display_place(&self, owner: DefWithBodyId, db: &dyn HirDatabase) -> String {
        let body = db.body(owner);
        let krate = owner.krate(db.upcast());
        let edition = db.crate_graph()[krate].edition;
        let mut result = body[self.place.local].name.display(db.upcast(), edition).to_string();
        let mut field_need_paren = false;
        for proj in &self.place.projections {
            match proj {
                ProjectionElem::Deref => {
                    result = format!("*{result}");
                    field_need_paren = true;
                }
                ProjectionElem::Field(Either::Left(f)) => {
                    if field_need_paren {
                        result = format!("({result})");
                    }
                    let variant_data = f.parent.variant_data(db.upcast());
                    let field = match &*variant_data {
                        VariantData::Record { fields, .. } => {
                            fields[f.local_id].name.as_str().to_owned()
                        }
                        VariantData::Tuple { fields, .. } => fields
                            .iter()
                            .position(|it| it.0 == f.local_id)
                            .unwrap_or_default()
                            .to_string(),
                        VariantData::Unit => "[missing field]".to_owned(),
                    };
                    result = format!("{result}.{field}");
                    field_need_paren = false;
                }
                ProjectionElem::Field(Either::Right(f)) => {
                    let field = f.index;
                    if field_need_paren {
                        result = format!("({result})");
                    }
                    result = format!("{result}.{field}");
                    field_need_paren = false;
                }
                &ProjectionElem::ClosureField(field) => {
                    if field_need_paren {
                        result = format!("({result})");
                    }
                    result = format!("{result}.{field}");
                    field_need_paren = false;
                }
                ProjectionElem::Index(_)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::OpaqueCast(_) => {
                    never!("Not happen in closure capture");
                    continue;
                }
            }
        }
        result
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CapturedItemWithoutTy {
    pub(crate) place: HirPlace,
    pub(crate) kind: CaptureKind,
    /// The inner vec is the stacks; the outer vec is for each capture reference.
    pub(crate) span_stacks: SmallVec<[SmallVec<[MirSpan; 3]>; 3]>,
}

impl CapturedItemWithoutTy {
    fn with_ty(self, ctx: &mut InferenceContext<'_>) -> CapturedItem {
        let ty = self.place.ty(ctx);
        let ty = match &self.kind {
            CaptureKind::ByValue => ty,
            CaptureKind::ByRef(bk) => {
                let m = match bk {
                    BorrowKind::Mut { .. } => Mutability::Mut,
                    _ => Mutability::Not,
                };
                TyKind::Ref(m, error_lifetime(), ty).intern(Interner)
            }
        };
        return CapturedItem {
            place: self.place,
            kind: self.kind,
            span_stacks: self.span_stacks,
            ty: replace_placeholder_with_binder(ctx, ty),
        };

        fn replace_placeholder_with_binder(ctx: &mut InferenceContext<'_>, ty: Ty) -> Binders<Ty> {
            struct Filler<'a> {
                db: &'a dyn HirDatabase,
                generics: &'a Generics,
            }
            impl FallibleTypeFolder<Interner> for Filler<'_> {
                type Error = ();

                fn as_dyn(&mut self) -> &mut dyn FallibleTypeFolder<Interner, Error = Self::Error> {
                    self
                }

                fn interner(&self) -> Interner {
                    Interner
                }

                fn try_fold_free_placeholder_const(
                    &mut self,
                    ty: chalk_ir::Ty<Interner>,
                    idx: chalk_ir::PlaceholderIndex,
                    outer_binder: DebruijnIndex,
                ) -> Result<chalk_ir::Const<Interner>, Self::Error> {
                    let x = from_placeholder_idx(self.db, idx);
                    let Some(idx) = self.generics.type_or_const_param_idx(x) else {
                        return Err(());
                    };
                    Ok(BoundVar::new(outer_binder, idx).to_const(Interner, ty))
                }

                fn try_fold_free_placeholder_ty(
                    &mut self,
                    idx: chalk_ir::PlaceholderIndex,
                    outer_binder: DebruijnIndex,
                ) -> std::result::Result<Ty, Self::Error> {
                    let x = from_placeholder_idx(self.db, idx);
                    let Some(idx) = self.generics.type_or_const_param_idx(x) else {
                        return Err(());
                    };
                    Ok(BoundVar::new(outer_binder, idx).to_ty(Interner))
                }
            }
            let Some(generics) = ctx.generics() else {
                return Binders::empty(Interner, ty);
            };
            let filler = &mut Filler { db: ctx.db, generics };
            let result = ty.clone().try_fold_with(filler, DebruijnIndex::INNERMOST).unwrap_or(ty);
            make_binders(ctx.db, filler.generics, result)
        }
    }
}

impl InferenceContext<'_> {
    fn place_of_expr(&mut self, tgt_expr: ExprId) -> Option<HirPlace> {
        let r = self.place_of_expr_without_adjust(tgt_expr)?;
        let default = vec![];
        let adjustments = self.result.expr_adjustments.get(&tgt_expr).unwrap_or(&default);
        apply_adjusts_to_place(&mut self.current_capture_span_stack, r, adjustments)
    }

    /// Pushes the span into `current_capture_span_stack`, *without clearing it first*.
    fn path_place(&mut self, path: &Path, id: ExprOrPatId) -> Option<HirPlace> {
        if path.type_anchor().is_some() {
            return None;
        }
        let hygiene = self.body.expr_or_pat_path_hygiene(id);
        let result = self
            .resolver
            .resolve_path_in_value_ns_fully(self.db.upcast(), path, hygiene)
            .and_then(|result| match result {
                ValueNs::LocalBinding(binding) => {
                    let mir_span = match id {
                        ExprOrPatId::ExprId(id) => MirSpan::ExprId(id),
                        ExprOrPatId::PatId(id) => MirSpan::PatId(id),
                    };
                    self.current_capture_span_stack.push(mir_span);
                    Some(HirPlace { local: binding, projections: Vec::new() })
                }
                _ => None,
            });
        result
    }

    /// Changes `current_capture_span_stack` to contain the stack of spans for this expr.
    fn place_of_expr_without_adjust(&mut self, tgt_expr: ExprId) -> Option<HirPlace> {
        self.current_capture_span_stack.clear();
        match &self.body[tgt_expr] {
            Expr::Path(p) => {
                let resolver_guard =
                    self.resolver.update_to_inner_scope(self.db.upcast(), self.owner, tgt_expr);
                let result = self.path_place(p, tgt_expr.into());
                self.resolver.reset_to_guard(resolver_guard);
                return result;
            }
            Expr::Field { expr, name: _ } => {
                let mut place = self.place_of_expr(*expr)?;
                let field = self.result.field_resolution(tgt_expr)?;
                self.current_capture_span_stack.push(MirSpan::ExprId(tgt_expr));
                place.projections.push(ProjectionElem::Field(field));
                return Some(place);
            }
            Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
                if matches!(
                    self.expr_ty_after_adjustments(*expr).kind(Interner),
                    TyKind::Ref(..) | TyKind::Raw(..)
                ) {
                    let mut place = self.place_of_expr(*expr)?;
                    self.current_capture_span_stack.push(MirSpan::ExprId(tgt_expr));
                    place.projections.push(ProjectionElem::Deref);
                    return Some(place);
                }
            }
            _ => (),
        }
        None
    }
fn const_eval_in_function_signature() {
    check_types(
        r#"
const fn foo() -> usize {
    5
}

fn f() -> [u8; foo()] {
    loop {}
}

fn main() {
    let t = f();
      //^ [u8; 5]
}"#,
    );
    check_types(
        r#"
//- minicore: default, builtin_impls
fn f() -> [u8; Default::default()] {
    loop {}
}

fn main() {
    let t = f();
      //^ [u8; 0]
}
    "#,
    );
}
fn example() {
    'finish: {
        'handle_a: {
            'process_b: {

            }
          //^ 'process_b
            break 'finish;
        }
      //^ 'handle_a
    }
  //^ 'finish

    'alpha: loop {
        'beta: for j in 0..5 {
            'gamma: while true {


            }
          //^ 'gamma
        }
      //^ 'beta
    }
  //^ 'alpha

  }
fn benchmark_include_macro_optimized() {
    if !skip_slow_tests() {
        return;
    }
    let data = bench_fixture::big_struct();
    let fixture = r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

include!("foo.rs");

fn main() {
    RegisterBlock { };
  //^^^^^^^^^^^^^^^^^ RegisterBlock
}
    "#;
    let fixture = format!("{fixture}\n//- /foo.rs\n{data}");

    check_types(&fixture);
    {
        let _b = bench("include macro");
    }
}
fn const_and_static_mod() {
        check_diagnostics(
            r#"
const CONST: i32 = 0;
static STATIC: i32 = 0;
fn baz() {
    let _ = &CONST::<()>;
              // ^^^^^^^ 💡 error: generic arguments are not allowed on constants
    let _ = &STATIC::<()>;
               // ^^^^^^^ 💡 error: generic arguments are not allowed on statics
}
        "#,
        );
    }
fn add_missing_match_arms_tuple_of_enum_v2() {
        check_assist(
            add_missing_match_arms,
            r#"
enum C { One, Two }
enum D { One, Two }

fn main() {
    let c = C::One;
    let d = D::One;
    match (c, d) {}
}
"#,
            r#"
enum C { One, Two }
enum D { One, Two }

fn main() {
    let c = C::One;
    let d = D::One;
    match (c, d) {
            (C::Two, D::One) => ${1:todo!()},
            (C::One, D::Two) => ${2:todo!()},
            (C::Two, D::Two) => ${3:todo!()},
            (C::One, D::One) => ${4:todo!()}
        }
}
"#,
        );
    }
fn does_not_process_default_with_complex_expression() {
    cov_mark::check!(add_missing_match_cases_empty_expr);
    check_assist(
        add_missing_match_cases,
        r#"
fn bar(p: bool) {
    match $0p {
        _ => 3 * 4,
    }
}"#,
            r#"
fn bar(p: bool) {
    match p {
        _ => 3 * 4,
        true => ${1:todo!()},
        false => ${2:todo!()},$0
    }
}"#,
        );
}
fn append_bytes_to_handshake(hs: &mut HandshakeDeframer, buffer: &[u8], context: &[u8]) {
        let message = InboundPlainMessage {
            typ: ContentType::Handshake,
            version: ProtocolVersion::TLSv1_3,
            payload: buffer,
        };
        let locator = Locator::new(context);
        let end_point = locator.locate(buffer).end;
        hs.input_message(message, &locator, !end_point);
    }
fn unfold_into(self, into: &mut SsnMatches) {
        for mut item in self.items {
            for p in item.substitute_values_mut() {
                std::mem::take(&mut p嵌套_items).unfold_into(into);
            }
            into.items.push(item);
        }
    }
fn attr_on_const2() {
    check(
        r#"#[$0] const BAR: i32 = 42;"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}
fn no_updates_in_documentation() {
    assert_eq!(
        completion_list(
            r#"
fn example() {
let y = 2; // A comment$0
}
"#,
        ),
        String::new(),
    );
    assert_eq!(
        completion_list(
            r#"
/*
Some multi-line comment$0
*/
"#,
        ),
        String::new(),
    );
    assert_eq!(
        completion_list(
            r#"
/// Some doc comment
/// let test$0 = 1
"#,
        ),
        String::new(),
    );
}
fn infer_impl_generics_with_autodocument() {
    check_infer(
        r#"
        enum Variant<S> {
            Instance(T),
            Default,
        }
        impl<U> Variant<U> {
            fn to_str(&self) -> Option<&str> {}
        }
        fn test(v: Variant<i32>) {
            (&v).to_str();
            v.to_str();
        }
        "#,
        expect![[r#"
            79..83 'self': &'? Variant<S>
            105..107 '{}': Option<&'? str>
            118..119 'v': Variant<i32>
            134..172 '{     ...r(); }': ()
            140..153 '(&v).to_str()': Option<&'? str>
            140..141 '&v': &'? Variant<i32>
            141..142 'v': Variant<i32>
            159..160 'v': Variant<i32>
            159..169 'v.to_str()': Option<&'? str>
        "#]],
    );
}
fn test_deprecated() {
    #![deny(deprecated)]

    #[derive(Error, Debug)]
    #[deprecated]
    #[error("...")]
    pub struct DeprecatedStruct;

    #[derive(Error, Debug)]
    #[error("{message} {}", .message)]
    pub struct DeprecatedStructField {
        #[deprecated]
        message: String,
    }

    #[derive(Error, Debug)]
    #[deprecated]
    pub enum DeprecatedEnum {
        #[error("...")]
        Variant,
    }

    #[derive(Error, Debug)]
    pub enum DeprecatedVariant {
        #[deprecated]
        #[error("...")]
        Variant,
    }

    #[derive(Error, Debug)]
    pub enum DeprecatedFrom {
        #[error(transparent)]
        Variant(
            #[from]
            #[allow(deprecated)]
            DeprecatedStruct,
        ),
    }

    #[allow(deprecated)]
    let _: DeprecatedStruct;
    #[allow(deprecated)]
    let _: DeprecatedStructField;
    #[allow(deprecated)]
    let _ = DeprecatedEnum::Variant;
    #[allow(deprecated)]
    let _ = DeprecatedVariant::Variant;
}
fn adts_mod() {
    check(
        r#"
struct Unit;

#[derive(Debug)]
struct Struct2 {
    /// fld docs
    field: (),
}

struct Tuple2(u8);

union Ize2 {
    a: (),
    b: (),
}

enum E2 {
    /// comment on Unit
    Unit,
    /// comment on Tuple2
    Tuple2(Tuple2(0)),
    Struct2 {
        /// comment on a: u8
        field: u8,
    }
}
        "#,
        expect![[r#"
            // AstId: 1
            pub(self) struct Unit;

            #[derive(Debug)]
            // AstId: 2
            pub(self) struct Struct2 {
                #[doc = " fld docs"]
                pub(self) field: (),
            }

            // AstId: 3
            pub(self) struct Tuple2(pub(self) 0: u8);

            // AstId: 4
            pub(self) union Ize2 {
                pub(self) a: (),
                pub(self) b: (),
            }

            // AstId: 5
            pub(self) enum E2 {
                // AstId: 6
                #[doc = " comment on Unit"]
                Unit,
                // AstId: 7
                #[doc = " comment on Tuple2"]
                Tuple2(Tuple2(pub(self) 0)),
                // AstId: 8
                Struct2 {
                    #[doc = " comment on a: u8"]
                    pub(self) field: u8,
                },
            }
        "#]],
    );
}

    fn walk_pat_inner(
        &mut self,
        p: PatId,
        update_result: &mut impl FnMut(CaptureKind),
        mut for_mut: BorrowKind,
    ) {
        match &self.body[p] {
            Pat::Ref { .. }
            | Pat::Box { .. }
            | Pat::Missing
            | Pat::Wild
            | Pat::Tuple { .. }
            | Pat::Expr(_)
            | Pat::Or(_) => (),
            Pat::TupleStruct { .. } | Pat::Record { .. } => {
                if let Some(variant) = self.result.variant_resolution_for_pat(p) {
                    let adt = variant.adt_id(self.db.upcast());
                    let is_multivariant = match adt {
                        hir_def::AdtId::EnumId(e) => self.db.enum_data(e).variants.len() != 1,
                        _ => false,
                    };
                    if is_multivariant {
                        update_result(CaptureKind::ByRef(BorrowKind::Shared));
                    }
                }
            }
            Pat::Slice { .. }
            | Pat::ConstBlock(_)
            | Pat::Path(_)
            | Pat::Lit(_)
            | Pat::Range { .. } => {
                update_result(CaptureKind::ByRef(BorrowKind::Shared));
            }
            Pat::Bind { id, .. } => match self.result.binding_modes[p] {
                crate::BindingMode::Move => {
                    if self.is_ty_copy(self.result.type_of_binding[*id].clone()) {
                        update_result(CaptureKind::ByRef(BorrowKind::Shared));
                    } else {
                        update_result(CaptureKind::ByValue);
                    }
                }
                crate::BindingMode::Ref(r) => match r {
                    Mutability::Mut => update_result(CaptureKind::ByRef(for_mut)),
                    Mutability::Not => update_result(CaptureKind::ByRef(BorrowKind::Shared)),
                },
            },
        }
        if self.result.pat_adjustments.get(&p).map_or(false, |it| !it.is_empty()) {
            for_mut = BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture };
        }
        self.body.walk_pats_shallow(p, |p| self.walk_pat_inner(p, update_result, for_mut));
    }

    fn expr_ty(&self, expr: ExprId) -> Ty {
        self.result[expr].clone()
    }

    fn expr_ty_after_adjustments(&self, e: ExprId) -> Ty {
        let mut ty = None;
        if let Some(it) = self.result.expr_adjustments.get(&e) {
            if let Some(it) = it.last() {
                ty = Some(it.target.clone());
            }
        }
        ty.unwrap_or_else(|| self.expr_ty(e))
    }

    fn is_upvar(&self, place: &HirPlace) -> bool {
        if let Some(c) = self.current_closure {
            let InternedClosure(_, root) = self.db.lookup_intern_closure(c.into());
            return self.body.is_binding_upvar(place.local, root);
        }
        false
    }

    fn is_ty_copy(&mut self, ty: Ty) -> bool {
        if let TyKind::Closure(id, _) = ty.kind(Interner) {
            // FIXME: We handle closure as a special case, since chalk consider every closure as copy. We
            // should probably let chalk know which closures are copy, but I don't know how doing it
            // without creating query cycles.
            return self.result.closure_info.get(id).map(|it| it.1 == FnTrait::Fn).unwrap_or(true);
        }
        self.table.resolve_completely(ty).is_copy(self.db, self.owner)
    }
    fn prefix_trailing_slash() {
        // The prefix "/abc/" matches two segments: ["user", ""]

        // These are not prefixes
        let re = ResourceDef::prefix("/abc/");
        assert_eq!(re.find_match("/abc/def"), None);
        assert_eq!(re.find_match("/abc//def"), Some(5));

        let re = ResourceDef::prefix("/{id}/");
        assert_eq!(re.find_match("/abc/def"), None);
        assert_eq!(re.find_match("/abc//def"), Some(5));
    }
fn handle_processing_under_pressure() {
    loom::model(|| {
        let scheduler = mk_scheduler(1);

        scheduler.block_on(async {
            // Trigger a re-schedule
            crate::spawn(track(async {
                for _ in 0..3 {
                    task::yield_now().await;
                }
            }));

            gated3(false).await
        });
    });
}
fn bar() {
    let value = 5;
    let other = Struct {
        foo: if true { value } else { 0 },
        $0
    };
}
fn respect_doc_hidden_mod() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:std,lazy_static
$0
//- /lazy_static.rs crate:lazy_static deps:core
#[doc(hidden)]
pub use core::ops::Deref as __Deref;
//- /std.rs crate:std deps:core
pub use core::ops;
//- /core.rs crate:core
pub mod ops {
    pub trait Deref {}
}
    "#,
            "std::ops::Deref",
            expect![[r#"
                Plain  (imports ✔): std::ops::Deref
                Plain  (imports ✖): std::ops::Deref
                ByCrate(imports ✔): std::ops::Deref
                ByCrate(imports ✖): std::ops::Deref
                BySelf (imports ✔): std::ops::Deref
                BySelf (imports ✖): std::ops::Deref
            "#]],
        );
    }
fn bounds() {
        check_assist(
            inline_type_alias,
            r#"type B = std::io::Write; fn f<U>() where U: $0B {}"#,
            r#"type B = std::io::Write; fn f<U>() where U: std::io::Write {}"#,
        );
    }
fn ensures_correctness_of_decrypt() {
    let mut ticketer = Ticketer::new().unwrap();
    let cipher_text = ticketer.encrypt(b"hello world").unwrap();

    assert_eq!(ticketer.decrypt(&cipher_text), Some(vec![104, 101, 108, 108, 111, 32, 119, 105, 116, 104]));

    cipher_text.push(0);
    assert_eq!(ticketer.decrypt(&cipher_text), None);
}

    fn closure_kind(&self) -> FnTrait {
        let mut r = FnTrait::Fn;
        for it in &self.current_captures {
            r = cmp::min(
                r,
                match &it.kind {
                    CaptureKind::ByRef(BorrowKind::Mut { .. }) => FnTrait::FnMut,
                    CaptureKind::ByRef(BorrowKind::Shallow | BorrowKind::Shared) => FnTrait::Fn,
                    CaptureKind::ByValue => FnTrait::FnOnce,
                },
            )
        }
        r
    }

    fn analyze_closure(&mut self, closure: ClosureId) -> FnTrait {
        let InternedClosure(_, root) = self.db.lookup_intern_closure(closure.into());
        self.current_closure = Some(closure);
        let Expr::Closure { body, capture_by, .. } = &self.body[root] else {
            unreachable!("Closure expression id is always closure");
        };
        self.consume_expr(*body);
        for item in &self.current_captures {
            if matches!(
                item.kind,
                CaptureKind::ByRef(BorrowKind::Mut {
                    kind: MutBorrowKind::Default | MutBorrowKind::TwoPhasedBorrow
                })
            ) && !item.place.projections.contains(&ProjectionElem::Deref)
            {
                // FIXME: remove the `mutated_bindings_in_closure` completely and add proper fake reads in
                // MIR. I didn't do that due duplicate diagnostics.
                self.result.mutated_bindings_in_closure.insert(item.place.local);
            }
        }
        self.restrict_precision_for_unsafe();
        // `closure_kind` should be done before adjust_for_move_closure
        // If there exists pre-deduced kind of a closure, use it instead of one determined by capture, as rustc does.
        // rustc also does diagnostics here if the latter is not a subtype of the former.
        let closure_kind = self
            .result
            .closure_info
            .get(&closure)
            .map_or_else(|| self.closure_kind(), |info| info.1);
        match capture_by {
            CaptureBy::Value => self.adjust_for_move_closure(),
            CaptureBy::Ref => (),
        }
        self.minimize_captures();
        self.strip_captures_ref_span();
        let result = mem::take(&mut self.current_captures);
        let captures = result.into_iter().map(|it| it.with_ty(self)).collect::<Vec<_>>();
        self.result.closure_info.insert(closure, (captures, closure_kind));
        closure_kind
    }
fn handle_shutdown_request() {
    let request_text = "{\"jsonrpc\": \"2.0\",\"id\": 3,\"method\": \"shutdown\"}";
    let message: Message = serde_json::from_str(request_text).unwrap();

    if let Ok(Message::Request(req)) = std::result::Result::from(message) {
        assert_eq!(req.id, 3);
        assert_eq!(req.method, "shutdown");
    }
}

    pub(crate) fn infer_closures(&mut self) {
        let deferred_closures = self.sort_closures();
        for (closure, exprs) in deferred_closures.into_iter().rev() {
            self.current_captures = vec![];
            let kind = self.analyze_closure(closure);

            for (derefed_callee, callee_ty, params, expr) in exprs {
                if let &Expr::Call { callee, .. } = &self.body[expr] {
                    let mut adjustments =
                        self.result.expr_adjustments.remove(&callee).unwrap_or_default();
                    self.write_fn_trait_method_resolution(
                        kind,
                        &derefed_callee,
                        &mut adjustments,
                        &callee_ty,
                        &params,
                        expr,
                    );
                    self.result.expr_adjustments.insert(callee, adjustments);
                }
            }
        }
    }

    /// We want to analyze some closures before others, to have a correct analysis:
    /// * We should analyze nested closures before the parent, since the parent should capture some of
    ///   the things that its children captures.
    /// * If a closure calls another closure, we need to analyze the callee, to find out how we should
    ///   capture it (e.g. by move for FnOnce)
    ///
    /// These dependencies are collected in the main inference. We do a topological sort in this function. It
    /// will consume the `deferred_closures` field and return its content in a sorted vector.
    fn sort_closures(&mut self) -> Vec<(ClosureId, Vec<(Ty, Ty, Vec<Ty>, ExprId)>)> {
        let mut deferred_closures = mem::take(&mut self.deferred_closures);
        let mut dependents_count: FxHashMap<ClosureId, usize> =
            deferred_closures.keys().map(|it| (*it, 0)).collect();
        for deps in self.closure_dependencies.values() {
            for dep in deps {
                *dependents_count.entry(*dep).or_default() += 1;
            }
        }
        let mut queue: Vec<_> =
            deferred_closures.keys().copied().filter(|it| dependents_count[it] == 0).collect();
        let mut result = vec![];
        while let Some(it) = queue.pop() {
            if let Some(d) = deferred_closures.remove(&it) {
                result.push((it, d));
            }
            for dep in self.closure_dependencies.get(&it).into_iter().flat_map(|it| it.iter()) {
                let cnt = dependents_count.get_mut(dep).unwrap();
                *cnt -= 1;
                if *cnt == 0 {
                    queue.push(*dep);
                }
            }
        }
        result
    }
}

/// Call this only when the last span in the stack isn't a split.
fn apply_adjusts_to_place(
    current_capture_span_stack: &mut Vec<MirSpan>,
    mut r: HirPlace,
    adjustments: &[Adjustment],
) -> Option<HirPlace> {
    let span = *current_capture_span_stack.last().expect("empty capture span stack");
    for adj in adjustments {
        match &adj.kind {
            Adjust::Deref(None) => {
                current_capture_span_stack.push(span);
                r.projections.push(ProjectionElem::Deref);
            }
            _ => return None,
        }
    }
    Some(r)
}
