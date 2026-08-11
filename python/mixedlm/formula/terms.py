from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InterceptTerm:
    pass


@dataclass(frozen=True)
class VariableTerm:
    name: str


@dataclass(frozen=True)
class InteractionTerm:
    variables: tuple[str, ...]

    @property
    def order(self) -> int:
        return len(self.variables)


@dataclass(frozen=True)
class FixedTerm:
    terms: tuple[InterceptTerm | VariableTerm | InteractionTerm, ...]
    has_intercept: bool = True


@dataclass(frozen=True)
class RandomTerm:
    expr: tuple[InterceptTerm | VariableTerm | InteractionTerm, ...]
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


@dataclass
class Formula:
    response: str
    fixed: FixedTerm
    random: tuple[RandomTerm, ...] = field(default_factory=tuple)
    response_denominator: str | None = None

    @property
    def response_expression(self) -> str:
        if self.response_denominator is None:
            return _format_identifier(self.response)
        return (
            f"{_format_identifier(self.response)} / {_format_identifier(self.response_denominator)}"
        )

    @property
    def fixed_variables(self) -> set[str]:
        result: set[str] = set()
        for term in self.fixed.terms:
            if isinstance(term, VariableTerm):
                result.add(term.name)
            elif isinstance(term, InteractionTerm):
                result.update(term.variables)
        return result

    @property
    def random_variables(self) -> set[str]:
        result: set[str] = set()
        for rterm in self.random:
            for term in rterm.expr:
                if isinstance(term, VariableTerm):
                    result.add(term.name)
                elif isinstance(term, InteractionTerm):
                    result.update(term.variables)
        return result

    @property
    def grouping_factors(self) -> set[str]:
        result: set[str] = set()
        for rterm in self.random:
            result.update(rterm.grouping_factors)
        return result

    @property
    def all_variables(self) -> set[str]:
        response_variables = {self.response}
        if self.response_denominator is not None:
            response_variables.add(self.response_denominator)
        return (
            response_variables
            | self.fixed_variables
            | self.random_variables
            | self.grouping_factors
        )

    def __str__(self) -> str:
        fixed_str = _format_fixed(self.fixed)
        random_strs = [_format_random(r) for r in self.random]
        rhs = " + ".join([fixed_str] + random_strs)
        return f"{self.response_expression} ~ {rhs}"


def _format_identifier(name: str) -> str:
    is_bare = bool(name) and name != "." and (name[0].isalpha() or name[0] in "_.")
    is_bare = is_bare and all(ch.isalnum() or ch in "_." for ch in name[1:])
    if is_bare:
        return name

    escaped = name.replace("\\", "\\\\").replace("`", "\\`")
    return f"`{escaped}`"


def _format_term(term: InterceptTerm | VariableTerm | InteractionTerm) -> str:
    if isinstance(term, InterceptTerm):
        return "1"
    elif isinstance(term, VariableTerm):
        return _format_identifier(term.name)
    else:
        return ":".join(_format_identifier(variable) for variable in term.variables)


def _format_fixed(fixed: FixedTerm) -> str:
    parts: list[str] = []
    if not fixed.has_intercept:
        parts.append("0")
    for term in fixed.terms:
        parts.append(_format_term(term))
    return " + ".join(parts) if parts else "1"


def _format_random(random: RandomTerm) -> str:
    expr_parts = [_format_term(t) for t in random.expr if not isinstance(t, InterceptTerm)]
    intercept = "1" if random.has_intercept else "0"
    expr_str = " + ".join([intercept, *expr_parts])

    group_str = _format_grouping(random.grouping)

    bar = "|" if random.correlated else "||"
    return f"({expr_str} {bar} {group_str})"


def _format_grouping(grouping: str | tuple[str, ...]) -> str:
    groups = grouping if isinstance(grouping, tuple) else (grouping,)
    return "/".join(_format_identifier(group) for group in groups)
