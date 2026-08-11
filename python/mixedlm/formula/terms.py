from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class InteractionTerm:
    variables: tuple[str | PowerTerm, ...]

    @property
    def order(self) -> int:
        return len(self.variables)

    @property
    def source_variables(self) -> tuple[str, ...]:
        return tuple(
            factor.name if isinstance(factor, PowerTerm) else factor for factor in self.variables
        )


@dataclass(frozen=True)
class FixedTerm:
    terms: tuple[InterceptTerm | VariableTerm | PowerTerm | InteractionTerm, ...]
    has_intercept: bool = True


@dataclass(frozen=True)
class RandomTerm:
    expr: tuple[InterceptTerm | VariableTerm | PowerTerm | InteractionTerm, ...]
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

    @property
    def fixed_variables(self) -> set[str]:
        result: set[str] = set()
        for term in self.fixed.terms:
            if isinstance(term, VariableTerm | PowerTerm):
                result.add(term.name)
            elif isinstance(term, InteractionTerm):
                result.update(term.source_variables)
        return result

    @property
    def random_variables(self) -> set[str]:
        result: set[str] = set()
        for rterm in self.random:
            for term in rterm.expr:
                if isinstance(term, VariableTerm | PowerTerm):
                    result.add(term.name)
                elif isinstance(term, InteractionTerm):
                    result.update(term.source_variables)
        return result

    @property
    def grouping_factors(self) -> set[str]:
        result: set[str] = set()
        for rterm in self.random:
            result.update(rterm.grouping_factors)
        return result

    @property
    def all_variables(self) -> set[str]:
        return (
            {self.response} | self.fixed_variables | self.random_variables | self.grouping_factors
        )

    def __str__(self) -> str:
        fixed_str = _format_fixed(self.fixed)
        random_strs = [_format_random(r) for r in self.random]
        rhs = " + ".join([fixed_str] + random_strs)
        return f"{_format_identifier(self.response)} ~ {rhs}"


def _format_identifier(name: str) -> str:
    is_bare = bool(name) and name != "." and (name[0].isalpha() or name[0] in "_.")
    is_bare = is_bare and all(ch.isalnum() or ch in "_." for ch in name[1:])
    if is_bare:
        return name

    escaped = name.replace("\\", "\\\\").replace("`", "\\`")
    return f"`{escaped}`"


def _format_factor(factor: str | PowerTerm) -> str:
    if isinstance(factor, PowerTerm):
        return f"I({_format_identifier(factor.name)}**{factor.exponent})"
    return _format_identifier(factor)


def format_term(term: InterceptTerm | VariableTerm | PowerTerm | InteractionTerm) -> str:
    if isinstance(term, InterceptTerm):
        return "1"
    elif isinstance(term, VariableTerm):
        return _format_identifier(term.name)
    elif isinstance(term, PowerTerm):
        return _format_factor(term)
    else:
        return ":".join(_format_factor(factor) for factor in term.variables)


def _format_term(term: InterceptTerm | VariableTerm | PowerTerm | InteractionTerm) -> str:
    return format_term(term)


def _format_fixed(fixed: FixedTerm) -> str:
    parts: list[str] = []
    if not fixed.has_intercept:
        parts.append("0")
    for term in fixed.terms:
        parts.append(format_term(term))
    return " + ".join(parts) if parts else "1"


def _format_random(random: RandomTerm) -> str:
    expr_parts = [format_term(t) for t in random.expr if not isinstance(t, InterceptTerm)]
    intercept = "1" if random.has_intercept else "0"
    expr_str = " + ".join([intercept, *expr_parts])

    group_str = _format_grouping(random.grouping)

    bar = "|" if random.correlated else "||"
    return f"({expr_str} {bar} {group_str})"


def _format_grouping(grouping: str | tuple[str, ...]) -> str:
    groups = grouping if isinstance(grouping, tuple) else (grouping,)
    return "/".join(_format_identifier(group) for group in groups)
