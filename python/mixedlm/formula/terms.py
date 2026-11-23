from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, assert_never


@dataclass(frozen=True)
class InterceptTerm:
    pass


@dataclass(frozen=True)
class VariableTerm:
    name: str


@dataclass(frozen=True)
class PowerTerm:
    name: str
    exponent: int

    def __post_init__(self) -> None:
        if not isinstance(self.exponent, int) or isinstance(self.exponent, bool):
            raise TypeError("Power exponents must be integers")
        if self.exponent < 0:
            raise ValueError("Power exponents must be non-negative")


# Expression nodes for computed formula variables such as ``log(x + 1)``.
# They are immutable and hashable so that terms built from them can be
# deduplicated and compared like plain variable names.


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Literal:
    value: bool | str


@dataclass(frozen=True)
class Unary:
    op: str
    operand: Expr


@dataclass(frozen=True)
class Binary:
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Call:
    """A formula function call with arguments bound to its signature.

    ``args`` holds the data expression. ``options`` holds the named options that
    differ from their defaults, in signature order, so equivalent spellings such
    as ``poly(x, 2)`` and ``poly(x, degree = 2)`` compare equal.
    """

    func: str
    args: tuple[Expr, ...]
    options: tuple[tuple[str, Expr], ...] = ()


Expr = Var | Num | Literal | Unary | Binary | Call

_REQUIRED: Any = object()


@dataclass(frozen=True)
class TransformParam:
    name: str
    kind: str
    default: Any = _REQUIRED
    positional: bool = False
    always: bool = False


@dataclass(frozen=True)
class TransformSpec:
    kind: str
    params: tuple[TransformParam, ...]

    @property
    def nestable(self) -> bool:
        """Elementwise functions may appear inside other expressions."""
        return self.kind in ("elementwise", "identity")


_DATA = TransformParam("x", "data")
_ELEMENTWISE_FUNCTIONS = (
    "log2",
    "log10",
    "log1p",
    "exp",
    "expm1",
    "sqrt",
    "abs",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "floor",
    "ceiling",
    "trunc",
    "sign",
)

TRANSFORM_FUNCTIONS: dict[str, TransformSpec] = {
    **{name: TransformSpec("elementwise", (_DATA,)) for name in _ELEMENTWISE_FUNCTIONS},
    "log": TransformSpec("elementwise", (_DATA, TransformParam("base", "number", math.e))),
    "I": TransformSpec("identity", (_DATA,)),
    "scale": TransformSpec(
        "scale",
        (
            _DATA,
            TransformParam("center", "logical", True),
            TransformParam("scale", "logical", True),
        ),
    ),
    "poly": TransformSpec(
        "poly",
        (
            _DATA,
            TransformParam("degree", "integer", 1, positional=True, always=True),
            TransformParam("raw", "logical", False),
        ),
    ),
    "factor": TransformSpec("factor", (_DATA,)),
    "as.factor": TransformSpec("factor", (_DATA,)),
    "C": TransformSpec(
        "factor", (_DATA, TransformParam("contr", "contrast", None, positional=True))
    ),
    "relevel": TransformSpec("factor", (_DATA, TransformParam("ref", "level"))),
}


@dataclass(frozen=True)
class TransformTerm:
    """A predictor computed from data columns, such as ``log(x)`` or ``poly(x, 2)``.

    Stateful transforms (``scale``, orthogonal ``poly``, and factor coding) are
    evaluated against the fitting data; predictions for new data reuse the
    fitted centering, scaling, polynomial basis, and levels.
    """

    expr: Call

    @property
    def func(self) -> str:
        return self.expr.func

    @property
    def spec(self) -> TransformSpec:
        return TRANSFORM_FUNCTIONS[self.expr.func]

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def is_categorical(self) -> bool:
        return self.kind == "factor"

    @cached_property
    def label(self) -> str:
        return format_expr(self.expr)

    @cached_property
    def variables(self) -> tuple[str, ...]:
        return expr_variables(self.expr)

    @property
    def data_expr(self) -> Expr:
        return self.expr.args[0]

    def option(self, name: str) -> Any:
        """Return an option's Python value, or its declared default."""
        for key, value in self.expr.options:
            if key == name:
                return _literal_value(value)
        for param in self.spec.params:
            if param.name == name:
                return param.default
        raise KeyError(name)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Names of the numeric columns this term produces."""
        if self.kind == "poly":
            return tuple(f"{self.label}{i}" for i in range(1, self.option("degree") + 1))
        return (self.label,)


Factor = str | PowerTerm | TransformTerm


@dataclass(frozen=True)
class InteractionTerm:
    variables: tuple[Factor, ...]

    @property
    def order(self) -> int:
        return len(self.variables)

    @property
    def source_variables(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(name for f in self.variables for name in factor_sources(f)))


ParsedTerm = InterceptTerm | VariableTerm | PowerTerm | TransformTerm | InteractionTerm


@dataclass(frozen=True)
class FixedTerm:
    terms: tuple[ParsedTerm, ...]
    has_intercept: bool = True


@dataclass(frozen=True)
class RandomTerm:
    expr: tuple[ParsedTerm, ...]
    grouping: str | tuple[str, ...]
    correlated: bool = True
    has_intercept: bool = True
    cov_type: str = "us"

    @property
    def is_nested(self) -> bool:
        return isinstance(self.grouping, tuple) and len(self.grouping) > 1

    @property
    def grouping_factors(self) -> tuple[str, ...]:
        if isinstance(self.grouping, str):
            return (self.grouping,)
        return self.grouping


def factor_sources(factor: Factor) -> tuple[str, ...]:
    """Return the data columns read by one interaction factor."""
    if isinstance(factor, str):
        return (factor,)
    if isinstance(factor, PowerTerm):
        return (factor.name,)
    if isinstance(factor, TransformTerm):
        return factor.variables
    assert_never(factor)


def term_sources(term: ParsedTerm) -> tuple[str, ...]:
    """Return the data columns read by a fixed or random-effect term."""
    if isinstance(term, InterceptTerm):
        return ()
    if isinstance(term, VariableTerm):
        return (term.name,)
    if isinstance(term, PowerTerm | TransformTerm):
        return factor_sources(term)
    if isinstance(term, InteractionTerm):
        return term.source_variables
    assert_never(term)


def term_factors(term: ParsedTerm) -> tuple[Factor, ...]:
    """Return the factors multiplied by a term."""
    if isinstance(term, InterceptTerm):
        return ()
    if isinstance(term, VariableTerm):
        return (term.name,)
    if isinstance(term, PowerTerm | TransformTerm):
        return (term,)
    if isinstance(term, InteractionTerm):
        return term.variables
    assert_never(term)


def expr_variables(expr: Expr) -> tuple[str, ...]:
    """Return the data columns an expression reads, in order of appearance."""
    names: dict[str, None] = {}

    def visit(node: Expr) -> None:
        if isinstance(node, Var):
            names[node.name] = None
        elif isinstance(node, Unary):
            visit(node.operand)
        elif isinstance(node, Binary):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, Call):
            # Options hold constants such as degrees, contrasts and levels.
            for arg in node.args:
                visit(arg)

    visit(expr)
    return tuple(names)


def _ordered_union(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(name for group in groups for name in group))


@dataclass
class Formula:
    response: str
    fixed: FixedTerm
    random: tuple[RandomTerm, ...] = field(default_factory=tuple)
    response_denominator: str | None = None
    # A computed response such as log(y), or the successes of cbind().
    response_expr: Expr | None = None
    # The failures of cbind(successes, failures).
    response_failures: Expr | None = None
    # A tuple default, unlike default_factory, gives the class attribute that
    # formulas pickled by earlier versions fall back to.
    offsets: tuple[Expr, ...] = ()
    # The text the formula was parsed from, which update() substitutes for '.' as
    # R does. It is not part of the formula's identity.
    source: str | None = field(default=None, compare=False, repr=False)

    @property
    def grouped_response(self) -> bool:
        """Whether the response gives binomial successes and trials."""
        return self.response_denominator is not None or self.response_failures is not None

    @property
    def plain_response(self) -> bool:
        """Whether the response is a single data column."""
        return self.response_expr is None and not self.grouped_response

    @property
    def response_expression(self) -> str:
        successes = (
            _format_identifier(self.response)
            if self.response_expr is None
            else format_expr(self.response_expr)
        )
        if self.response_failures is not None:
            return f"cbind({successes}, {format_expr(self.response_failures)})"
        if self.response_denominator is None:
            return successes
        return f"{successes} / {_format_identifier(self.response_denominator)}"

    @property
    def failures_label(self) -> str | None:
        """The model-frame column name of cbind() failures."""
        if self.response_failures is None:
            return None
        return format_expr(self.response_failures)

    @property
    def response_variables(self) -> set[str]:
        sources = (
            (self.response,) if self.response_expr is None else expr_variables(self.response_expr)
        )
        result = set(sources)
        if self.response_denominator is not None:
            result.add(self.response_denominator)
        if self.response_failures is not None:
            result.update(expr_variables(self.response_failures))
        return result

    @property
    def response_variables_ordered(self) -> tuple[str, ...]:
        """The response's data columns in order of appearance."""
        response = (
            (self.response,) if self.response_expr is None else expr_variables(self.response_expr)
        )
        denominator = (self.response_denominator,) if self.response_denominator else ()
        failures = (
            expr_variables(self.response_failures) if self.response_failures is not None else ()
        )
        return _ordered_union(response, denominator, failures)

    @property
    def offset_variables(self) -> set[str]:
        return {name for offset in self.offsets for name in expr_variables(offset)}

    @property
    def fixed_variables(self) -> set[str]:
        return {name for term in self.fixed.terms for name in term_sources(term)}

    @property
    def random_variables(self) -> set[str]:
        return {name for rterm in self.random for term in rterm.expr for name in term_sources(term)}

    @property
    def grouping_factors(self) -> set[str]:
        result: set[str] = set()
        for rterm in self.random:
            result.update(rterm.grouping_factors)
        return result

    @property
    def all_variables(self) -> set[str]:
        return (
            self.response_variables
            | self.fixed_variables
            | self.offset_variables
            | self.random_variables
            | self.grouping_factors
        )

    def model_variables(self) -> list[str]:
        """Return every data column the formula reads, in order of appearance."""
        response = (
            (self.response,) if self.response_expr is None else expr_variables(self.response_expr)
        )
        denominator = (self.response_denominator,) if self.response_denominator else ()
        failures = (
            expr_variables(self.response_failures) if self.response_failures is not None else ()
        )
        random: list[str] = []
        for rterm in self.random:
            random.extend(rterm.grouping_factors)
            for term in rterm.expr:
                random.extend(term_sources(term))
        return list(
            _ordered_union(
                response,
                denominator,
                failures,
                *(term_sources(term) for term in self.fixed.terms),
                *(expr_variables(offset) for offset in self.offsets),
                tuple(random),
            )
        )

    def __str__(self) -> str:
        fixed_str = _format_fixed(self.fixed)
        offset_strs = [f"offset({format_expr(offset)})" for offset in self.offsets]
        random_strs = [_format_random(r) for r in self.random]
        rhs = " + ".join([fixed_str, *offset_strs, *random_strs])
        return f"{self.response_expression} ~ {rhs}"


def _format_identifier(name: str) -> str:
    is_bare = bool(name) and name != "." and (name[0].isalpha() or name[0] in "_.")
    is_bare = is_bare and all(ch.isalnum() or ch in "_." for ch in name[1:])
    is_bare = is_bare and not (name[0] == "." and len(name) > 1 and name[1].isdigit())
    if is_bare:
        return name

    escaped = name.replace("\\", "\\\\").replace("`", "\\`")
    return f"`{escaped}`"


def _format_number(value: float) -> str:
    if math.isfinite(value) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


_BINARY_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "**": 4}
_UNARY_PRECEDENCE = 3
_ATOM_PRECEDENCE = 5


def _expr_precedence(expr: Expr) -> int:
    if isinstance(expr, Binary):
        return _BINARY_PRECEDENCE[expr.op]
    if isinstance(expr, Unary) or (isinstance(expr, Num) and expr.value < 0):
        return _UNARY_PRECEDENCE
    return _ATOM_PRECEDENCE


def _format_operand(expr: Expr, needs_parens: bool) -> str:
    text = format_expr(expr)
    return f"({text})" if needs_parens else text


def format_expr(expr: Expr) -> str:
    """Format an expression so that parsing the text gives the same expression."""
    if isinstance(expr, Var):
        return _format_identifier(expr.name)
    if isinstance(expr, Num):
        return _format_number(expr.value)
    if isinstance(expr, Literal):
        if isinstance(expr.value, bool):
            return "TRUE" if expr.value else "FALSE"
        escaped = expr.value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(expr, Unary):
        operand = _format_operand(expr.operand, _expr_precedence(expr.operand) < _UNARY_PRECEDENCE)
        return f"{expr.op}{operand}"
    if isinstance(expr, Binary):
        precedence = _BINARY_PRECEDENCE[expr.op]
        left_precedence = _expr_precedence(expr.left)
        right_precedence = _expr_precedence(expr.right)
        if expr.op == "**":
            # Right associative: (a**b)**c needs parentheses, a**b**c does not.
            left = _format_operand(expr.left, left_precedence <= precedence)
            right = _format_operand(expr.right, right_precedence < _UNARY_PRECEDENCE)
            return f"{left}**{right}"
        left = _format_operand(expr.left, left_precedence < precedence)
        right = _format_operand(expr.right, right_precedence <= precedence)
        return f"{left} {expr.op} {right}"
    if isinstance(expr, Call):
        parts = [format_expr(arg) for arg in expr.args]
        spec = TRANSFORM_FUNCTIONS.get(expr.func)
        params = {param.name: param for param in spec.params} if spec is not None else {}
        for name, value in expr.options:
            param = params.get(name)
            # Contrast names are stored as strings but spelled as bare names.
            if param is not None and param.kind == "contrast" and isinstance(value, Literal):
                text = str(value.value)
            else:
                text = format_expr(value)
            parts.append(text if param is not None and param.positional else f"{name} = {text}")
        return f"{_format_identifier(expr.func)}({', '.join(parts)})"
    assert_never(expr)


def _literal_value(value: Expr) -> Any:
    if isinstance(value, Num):
        return int(value.value) if value.value == int(value.value) else value.value
    if isinstance(value, Literal):
        return value.value
    raise TypeError(f"Expected a constant option, got {format_expr(value)}")


def _format_factor(factor: Factor) -> str:
    if isinstance(factor, PowerTerm):
        return f"I({_format_identifier(factor.name)}**{factor.exponent})"
    if isinstance(factor, TransformTerm):
        return factor.label
    return _format_identifier(factor)


def format_term(term: ParsedTerm) -> str:
    if isinstance(term, InterceptTerm):
        return "1"
    elif isinstance(term, VariableTerm):
        return _format_identifier(term.name)
    elif isinstance(term, PowerTerm | TransformTerm):
        return _format_factor(term)
    elif isinstance(term, InteractionTerm):
        return ":".join(_format_factor(factor) for factor in term.variables)
    else:
        assert_never(term)


def _format_term(term: ParsedTerm) -> str:
    return format_term(term)


def _printing_order(terms: tuple[ParsedTerm, ...]) -> list[ParsedTerm | _Introduce]:
    """Order terms for printing so that the printed formula parses to the same terms.

    An interaction lists its factors in their order of first appearance, as R's
    terms() does, so printing strictly by degree can change it: ``x:z + z`` would
    print as ``z + x:z``, which parses to ``z:x``. Terms print by degree except
    when a term would introduce a factor ahead of one that an interaction lists
    before it; then a later-degree term that introduces them in order goes
    first. Terms of each degree keep their order. When no order of the terms
    works, as for ``(x + f):k``, the factors are introduced in order by no-op
    pairs such as ``f - f``, which R's terms() also reads.
    """
    factors = [term_factors(term) for term in terms]
    pairs = [
        (earlier, later)
        for term_factors_ in factors
        for i, later in enumerate(term_factors_)
        for earlier in term_factors_[:i]
    ]
    queues: dict[int, list[int]] = {}
    for index, term_factors_ in enumerate(factors):
        queues.setdefault(len(term_factors_), []).append(index)
    seen: set[Factor] = set()
    order: list[ParsedTerm | _Introduce] = []

    def safe(index: int) -> bool:
        introduced: list[Factor] = []
        for factor in factors[index]:
            if factor in seen:
                continue
            if any(
                later == factor and earlier not in seen and earlier not in introduced
                for earlier, later in pairs
            ):
                return False
            introduced.append(factor)
        return True

    while any(queues.values()):
        heads = [queue[0] for _, queue in sorted(queues.items()) if queue]
        chosen = next((head for head in heads if safe(head)), None)
        if chosen is None:
            # Introduce a factor that blocks the first head, one that no unseen
            # factor must precede, so the head can then introduce the rest.
            head = list(factors[heads[0]])
            blockers = [
                earlier
                for earlier, later in pairs
                if later in head
                and later not in seen
                and earlier not in seen
                # The head itself introduces its earlier factors in order.
                and not (earlier in head and head.index(earlier) < head.index(later))
            ]
            ready = None
            for blocker in dict.fromkeys(blockers):
                # Walk back to a factor that no unseen factor must precede.
                factor: Factor = blocker
                visited: set[Factor] = set()
                while factor not in visited:
                    visited.add(factor)
                    earlier = [e for e, later in pairs if later == factor and e not in seen]
                    if not earlier:
                        ready = factor
                        break
                    factor = earlier[0]
                if ready is not None:
                    break
            if ready is not None:
                order.append(_Introduce(ready))
                seen.add(ready)
                continue
            chosen = heads[0]
        queues[len(factors[chosen])].pop(0)
        order.append(terms[chosen])
        seen.update(factors[chosen])
    return order


@dataclass(frozen=True)
class _Introduce:
    """A factor printed as ``f - f``, which only fixes its order of appearance."""

    factor: Factor


def _format_printed(item: ParsedTerm | _Introduce) -> str:
    if isinstance(item, _Introduce):
        name = format_term(
            VariableTerm(item.factor) if isinstance(item.factor, str) else item.factor
        )
        return f"{name} - {name}"
    return format_term(item)


def _format_fixed(fixed: FixedTerm) -> str:
    parts: list[str] = []
    if not fixed.has_intercept:
        parts.append("0")
    for item in _printing_order(fixed.terms):
        parts.append(_format_printed(item))
    return " + ".join(parts) if parts else "1"


def _format_random(random: RandomTerm) -> str:
    expr_parts = [
        _format_printed(t) for t in _printing_order(random.expr) if not isinstance(t, InterceptTerm)
    ]
    intercept = "1" if random.has_intercept else "0"
    expr_str = " + ".join([intercept, *expr_parts])

    group_str = _format_grouping(random.grouping)

    bar = "|" if random.correlated else "||"
    return f"({expr_str} {bar} {group_str})"


def _format_grouping(grouping: str | tuple[str, ...]) -> str:
    groups = grouping if isinstance(grouping, tuple) else (grouping,)
    return ":".join(_format_identifier(group) for group in groups)
