from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, assert_never

from mixedlm.formula.terms import (
    _REQUIRED,
    TRANSFORM_FUNCTIONS,
    Binary,
    Call,
    Expr,
    Factor,
    FixedTerm,
    Formula,
    InteractionTerm,
    InterceptTerm,
    Literal,
    Num,
    ParsedTerm,
    PowerTerm,
    RandomTerm,
    TransformParam,
    TransformTerm,
    Unary,
    Var,
    VariableTerm,
    _format_grouping,
    _format_random,
    _format_term,
    format_expr,
    term_factors,
)

TermMap = dict[frozenset[Factor], ParsedTerm]
# Intermediate products keep the first spelling of each factor set.
FactorMap = dict[frozenset[Factor], tuple[Factor, ...]]

_LOGICAL_NAMES = {"TRUE": True, "True": True, "T": True, "FALSE": False, "False": False, "F": False}
_CONTRAST_NAMES = ("treatment", "sum", "helmert", "poly", "deviation")
_SPECIAL_FUNCTIONS = ("offset", "cbind")
# Only ASCII digits form numbers; str.isdigit() also accepts superscripts.
_DIGITS = frozenset("0123456789")


class TokenType(Enum):
    TILDE = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    POWER = auto()
    CARET = auto()
    COLON = auto()
    SLASH = auto()
    PIPE = auto()
    DOUBLE_PIPE = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    EQUALS = auto()
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    position: int


_SINGLE_CHARACTER_TOKENS = {
    "~": TokenType.TILDE,
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "^": TokenType.CARET,
    ":": TokenType.COLON,
    "/": TokenType.SLASH,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    ",": TokenType.COMMA,
    "=": TokenType.EQUALS,
}


class Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def peek(self, offset: int = 0) -> str | None:
        index = self.pos + offset
        if index >= len(self.text):
            return None
        return self.text[index]

    def advance(self) -> str | None:
        ch = self.peek()
        self.pos += 1
        return ch

    def skip_whitespace(self) -> None:
        ch = self.peek()
        while ch is not None and ch in " \t\n\r":
            self.advance()
            ch = self.peek()

    def _quoted(self, quote: str, start_pos: int, kind: str) -> str:
        """Read a backslash-escaped quoted run after its opening quote."""
        text = ""
        while True:
            next_ch = self.advance()
            if next_ch is None:
                raise ValueError(f"Unterminated {kind} starting at position {start_pos}")
            if next_ch == quote:
                return text
            if next_ch == "\\":
                escaped = self.advance()
                if escaped is None:
                    raise ValueError(f"Unterminated {kind} starting at position {start_pos}")
                text += escaped
            else:
                text += next_ch

    def _number(self) -> str:
        start = self.pos
        while (ch := self.peek()) is not None and ch in _DIGITS:
            self.advance()
        following = self.peek(1) or ""
        # R and Python also accept a trailing point, as in 5. or 1.e3.
        if self.peek() == "." and (
            following in _DIGITS
            or following in ("e", "E")
            or not (following.isalnum() or following in ("_", "."))
        ):
            self.advance()
            while (ch := self.peek()) is not None and ch in _DIGITS:
                self.advance()
        if self.peek() in ("e", "E"):
            sign = 2 if self.peek(1) in ("+", "-") else 1
            if self.peek(sign) in _DIGITS:
                for _ in range(sign):
                    self.advance()
                while (ch := self.peek()) is not None and ch in _DIGITS:
                    self.advance()
        return self.text[start : self.pos]

    def tokenize(self) -> Iterator[Token]:
        while True:
            self.skip_whitespace()
            start_pos = self.pos
            ch = self.peek()

            if ch is None:
                yield Token(TokenType.EOF, "", start_pos)
                return

            if ch == "*":
                self.advance()
                if self.peek() == "*":
                    self.advance()
                    yield Token(TokenType.POWER, "**", start_pos)
                else:
                    yield Token(TokenType.STAR, "*", start_pos)
            elif ch == "|":
                self.advance()
                if self.peek() == "|":
                    self.advance()
                    yield Token(TokenType.DOUBLE_PIPE, "||", start_pos)
                else:
                    yield Token(TokenType.PIPE, "|", start_pos)
            elif ch in _SINGLE_CHARACTER_TOKENS:
                self.advance()
                yield Token(_SINGLE_CHARACTER_TOKENS[ch], ch, start_pos)
            elif ch == "`":
                self.advance()
                ident = self._quoted("`", start_pos, "quoted identifier")
                yield Token(TokenType.IDENTIFIER, ident, start_pos)
            elif ch in ("'", '"'):
                self.advance()
                yield Token(TokenType.STRING, self._quoted(ch, start_pos, "string"), start_pos)
            elif ch in _DIGITS or (ch == "." and self.peek(1) in _DIGITS):
                yield Token(TokenType.NUMBER, self._number(), start_pos)
            elif ch.isalpha() or ch == "_" or ch == ".":
                ident = ""
                next_ch = self.peek()
                while next_ch is not None and (next_ch.isalnum() or next_ch in "_."):
                    advanced = self.advance()
                    if advanced is None:
                        break
                    ident += advanced
                    next_ch = self.peek()
                yield Token(TokenType.IDENTIFIER, ident, start_pos)
            else:
                raise ValueError(f"Unexpected character '{ch}' at position {start_pos}")


def _interact(left: FactorMap, right: FactorMap) -> FactorMap:
    """Form every product of a left and a right term, using each factor once."""
    result: FactorMap = {}
    for left_key, left_factors in left.items():
        for right_key, right_factors in right.items():
            key = left_key | right_key
            if key not in result:
                result[key] = left_factors + tuple(f for f in right_factors if f not in left_key)
    return result


def _power(terms: FactorMap, power: int) -> FactorMap:
    """Expand ``(a + b)^k`` in the order of R's PowerTerms.

    Each step forms every product of a left term with a term of the previous
    result, left terms outermost, and keeps the first of any repeats, so
    ``(a + b)^2`` gives ``a, a:b, b``. The order shows when the result is crossed
    with other terms of mixed degree.
    """
    result = terms
    for _ in range(power - 1):
        product: FactorMap = {}
        for left_key, left_factors in terms.items():
            for right_key, right_factors in result.items():
                key = left_key | right_key
                if key not in product:
                    product[key] = left_factors + tuple(
                        f for f in right_factors if f not in left_key
                    )
        result = product
    return result


def _cross(left: FactorMap, right: FactorMap) -> FactorMap:
    """Expand ``a * b`` to ``a + b + a:b``."""
    result = dict(left)
    for key, factors in right.items():
        result.setdefault(key, factors)
    for key, factors in _interact(left, right).items():
        result.setdefault(key, factors)
    return result


def _nest(outer: FactorMap, inner: FactorMap) -> FactorMap:
    """Expand ``a / b`` to ``a + a:b``, nesting within every outer factor."""
    result = dict(outer)
    parent = tuple(dict.fromkeys(f for factors in outer.values() for f in factors))
    for key, factors in _interact({frozenset(parent): parent}, inner).items():
        result.setdefault(key, factors)
    return result


def _by_degree(terms: Iterable[ParsedTerm]) -> tuple[ParsedTerm, ...]:
    """Order terms by interaction degree, keeping their order within a degree, as R does."""
    return tuple(sorted(terms, key=lambda term: len(term_factors(term))))


def _in_variable_order(
    terms: Iterable[ParsedTerm], order: Mapping[Factor, int]
) -> list[ParsedTerm]:
    """Write each interaction's factors in their order of first appearance.

    R's terms() does this, so ``b + a:b`` has the term ``b:a``, whose first
    factor varies fastest in the design. ``order`` maps factors to the position
    where they first appear in the formula part.
    """
    ordered: list[ParsedTerm] = []
    for term in terms:
        if isinstance(term, InteractionTerm):
            factors = tuple(sorted(term.variables, key=lambda f: order.get(f, len(order))))
            term = InteractionTerm(factors) if factors != term.variables else term
        ordered.append(term)
    return ordered


def _correlated_block(term: ParsedTerm) -> bool:
    """Whether a term inside '||' keeps its own correlated block, as in lme4.

    lme4 expands ``(1 + poly(x, 2) || g)`` term by term, to ``(1 | g) +
    (0 + poly(x, 2) | g)``, so a term with several polynomial columns, including
    an interaction such as ``poly(x, 2):z``, stays correlated. Factor slopes stay
    independent column by column, as documented.
    """
    return any(
        isinstance(factor, TransformTerm)
        and factor.kind == "poly"
        and int(factor.option("degree")) > 1
        for factor in term_factors(term)
    )


def _split_uncorrelated(
    expr: tuple[ParsedTerm, ...], has_intercept: bool
) -> list[tuple[tuple[ParsedTerm, ...], bool, bool]]:
    """Split a '||' bar around its correlated blocks, keeping the term order."""
    parts: list[tuple[tuple[ParsedTerm, ...], bool, bool]] = []
    run: list[ParsedTerm] = []
    run_intercept = has_intercept
    for term in expr:
        if not _correlated_block(term):
            run.append(term)
            continue
        if run or run_intercept:
            parts.append((tuple(run), False, run_intercept))
        run, run_intercept = [], False
        parts.append(((term,), True, False))
    if run or run_intercept or not parts:
        parts.append((tuple(run), False, run_intercept))
    return parts


def _apply_terms(target: TermMap, terms: FactorMap, subtracting: bool) -> None:
    for key, factors in terms.items():
        if subtracting:
            target.pop(key, None)
        else:
            target.setdefault(key, Parser._factor_term(factors))


class Parser:
    def __init__(
        self, tokens: list[Token], *, dot_groups: bool = False, none: str | None = None
    ) -> None:
        self.tokens = tokens
        self.pos = 0
        # Whether a '.' is a group of terms, as the old right-hand side in an update
        # read as written; such a reading does not check its bars for columns.
        self._dot_groups = dot_groups
        # A name standing for no terms, as an old fixed part with none in an update.
        self._none = none
        # Where each factor first appears in the fixed part or the current bar.
        self._appearance: dict[Factor, int] = {}

    def current(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def advance(self) -> Token:
        tok = self.current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return tok

    def expect(self, type_: TokenType) -> Token:
        tok = self.current()
        if tok.type != type_:
            raise ValueError(f"Expected {type_}, got {tok.type} at position {tok.position}")
        return self.advance()

    def _at_call(self) -> bool:
        return self.peek().type == TokenType.IDENTIFIER and self.peek(1).type == TokenType.LPAREN

    def parse(self) -> Formula:
        response = self._parse_response()
        self.expect(TokenType.TILDE)
        fixed_terms, random_terms, has_intercept, offsets = self._parse_rhs()
        self.expect(TokenType.EOF)

        fixed_terms = _in_variable_order(fixed_terms, self._appearance)
        fixed = FixedTerm(terms=_by_degree(fixed_terms), has_intercept=has_intercept)
        return replace(response, fixed=fixed, random=tuple(random_terms), offsets=offsets)

    def _parse_response(self) -> Formula:
        empty = FixedTerm(terms=())
        if self._at_call():
            position = self.peek().position
            if self.peek().value == "cbind":
                self.advance()
                successes, failures = self._parse_call_arguments("cbind", position)
                response_expr = None if isinstance(successes, Var) else successes
                name = successes.name if isinstance(successes, Var) else format_expr(successes)
                return Formula(name, empty, response_expr=response_expr, response_failures=failures)
            spec = TRANSFORM_FUNCTIONS.get(self.peek().value)
            if spec is not None and not spec.nestable:
                raise ValueError(
                    f"{self.peek().value}() at position {position} cannot transform a "
                    "response; transform the response column before fitting"
                )
            expr = self._parse_arith()
            if not _expr_has_variable(expr):
                raise ValueError(
                    f"The response {format_expr(expr)} at position {position} must refer "
                    "to a data column"
                )
            if isinstance(expr, Call) and expr.func == "I" and isinstance(expr.args[0], Var):
                # I(y) is the same response as y.
                return Formula(expr.args[0].name, empty)
            return Formula(format_expr(expr), empty, response_expr=expr)

        tok = self.expect(TokenType.IDENTIFIER)
        if self.peek().type != TokenType.SLASH:
            return Formula(tok.value, empty)

        self.advance()
        denominator = self.expect(TokenType.IDENTIFIER)
        return Formula(tok.value, empty, response_denominator=denominator.value)

    def _parse_rhs(
        self,
    ) -> tuple[list[ParsedTerm], list[RandomTerm], bool, tuple[Expr, ...]]:
        fixed_terms: TermMap = {}
        random_terms: list[RandomTerm] = []
        offsets: list[Expr] = []
        has_intercept = True

        first = True
        while True:
            subtracting = False
            if self.peek().type in (TokenType.PLUS, TokenType.MINUS):
                subtracting = self.advance().type == TokenType.MINUS
            elif not first:
                if self.peek().type == TokenType.EOF:
                    break
                self._unexpected("'+' or '-' between terms")

            if self.peek().type == TokenType.LPAREN and self._is_random_group():
                parsed_random, empty_at = self._parse_random_term()
                if empty_at is not None and not subtracting and not self._dot_groups:
                    raise ValueError(
                        f"The random-effect term at position {empty_at} has no intercept or slopes"
                    )
                if subtracting:
                    removed = {_bar_key(term) for term in parsed_random}
                    random_terms = [term for term in random_terms if _bar_key(term) not in removed]
                else:
                    random_terms.extend(parsed_random)
            elif self._at_call() and self.peek().value == "offset":
                position = self.advance().position
                (offset,) = self._parse_call_arguments("offset", position)
                if subtracting:
                    offsets = [existing for existing in offsets if existing != offset]
                elif offset not in offsets:
                    offsets.append(offset)
            elif self.peek().type in (TokenType.IDENTIFIER, TokenType.NUMBER, TokenType.LPAREN):
                has_intercept = self._parse_effect(fixed_terms, has_intercept, subtracting)
            else:
                self._unexpected("a formula term")
            first = False

        return list(fixed_terms.values()), random_terms, has_intercept, tuple(offsets)

    def _is_random_group(self) -> bool:
        """Look ahead from '(' for a bar at the same nesting depth."""
        depth = 0
        index = self.pos
        while True:
            token = self.tokens[index]
            if token.type == TokenType.LPAREN:
                depth += 1
            elif token.type == TokenType.RPAREN:
                depth -= 1
                if depth == 0:
                    return False
            elif token.type in (TokenType.PIPE, TokenType.DOUBLE_PIPE) and depth == 1:
                return True
            elif token.type == TokenType.EOF:
                # Let the random-term parser report the unterminated group.
                return True
            index += 1

    def _unexpected(self, expected: str) -> None:
        token = self.peek()
        raise ValueError(f"Expected {expected}, got {token.type.name} at position {token.position}")

    def _parse_effect(self, target: TermMap, has_intercept: bool, subtracting: bool) -> bool:
        """Apply one additive term, expanding R's formula operators."""
        if self.peek().type == TokenType.NUMBER:
            tok = self.advance()
            if float(tok.value) not in (0.0, 1.0):
                raise ValueError(
                    f"Unexpected number {tok.value} at position {tok.position}; only 0 and 1, "
                    "which remove or keep the intercept, can be formula terms"
                )
            return (float(tok.value) == 1.0) != subtracting

        _apply_terms(target, self._parse_product(), subtracting)
        return has_intercept

    def _parse_product(self) -> FactorMap:
        """Parse ``*`` (crossing) and ``/`` (nesting), which bind less tightly than ``:``."""
        terms = self._parse_interaction_terms()
        while self.peek().type in (TokenType.STAR, TokenType.SLASH):
            nesting = self.advance().type == TokenType.SLASH
            right = self._parse_interaction_terms()
            terms = _nest(terms, right) if nesting else _cross(terms, right)
        return terms

    def _parse_interaction_terms(self) -> FactorMap:
        terms = self._parse_crossing_power()
        while self.peek().type == TokenType.COLON:
            self.advance()
            terms = _interact(terms, self._parse_crossing_power())
        return terms

    def _parse_crossing_power(self) -> FactorMap:
        """Parse ``(a + b)^k``: every interaction of the terms up to order ``k``."""
        token = self.peek()
        parenthesized = token.type == TokenType.LPAREN or (
            self._dot_groups and token.type == TokenType.IDENTIFIER and token.value == "."
        )
        terms = self._parse_term_atom()
        if self.peek().type not in (TokenType.CARET, TokenType.POWER):
            return terms
        operator = self.advance()
        token = self.peek()
        power = float(token.value) if token.type == TokenType.NUMBER else 0.0
        if power < 1 or power != int(power):
            self._unexpected("a positive integer power")
        self.advance()
        # A parenthesized group crosses its terms, as R reads (x)^2; only a bare
        # x^2, which R silently reads as x, is likely meant as a power.
        if not parenthesized and len(terms) == 1 and len(next(iter(terms))) == 1 and power > 1:
            raise ValueError(
                f"'{operator.value}' at position {operator.position} crosses formula terms, "
                "as in (a + b)^2; use I(x**2) or poly(x, 2) for a power of one predictor"
            )
        return _power(terms, int(power))

    def _parse_term_atom(self) -> FactorMap:
        token = self.peek()
        if token.type == TokenType.IDENTIFIER and token.value == self._none and not self._at_call():
            self.advance()
            return {}
        if token.type != TokenType.LPAREN:
            factor = self._parse_factor()
            return {frozenset((factor,)): (factor,)}

        self.advance()
        group: TermMap = {}
        first = True
        while first or self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            subtracting = False
            if self.peek().type in (TokenType.PLUS, TokenType.MINUS):
                subtracting = self.advance().type == TokenType.MINUS
            if self.peek().type not in (TokenType.IDENTIFIER, TokenType.LPAREN):
                self._unexpected("a variable or parenthesized formula terms")
            _apply_terms(group, self._parse_product(), subtracting)
            first = False
        self.expect(TokenType.RPAREN)
        return {frozenset(term_factors(term)): term_factors(term) for term in group.values()}

    def _parse_factor(self) -> Factor:
        factor = self._parse_factor_atom()
        self._appearance.setdefault(factor, len(self._appearance))
        return factor

    def _parse_factor_atom(self) -> Factor:
        if self.peek().type != TokenType.IDENTIFIER:
            self._unexpected("a variable or function call")
        if not self._at_call():
            return self.advance().value

        tok = self.advance()
        func = tok.value
        if func in _SPECIAL_FUNCTIONS:
            place = "as a response" if func == "cbind" else "as a separate additive term"
            raise ValueError(f"{func}() at position {tok.position} is only allowed {place}")
        call = self._parse_call(func, tok.position, nested=False)
        if func == "I":
            power = _as_power_term(call.args[0])
            if power is not None:
                return power
        return TransformTerm(call)

    # Arithmetic inside function calls follows Python precedence:
    # unary signs bind less tightly than powers, so -x**2 is -(x**2).

    def _parse_arith(self) -> Expr:
        left = self._parse_arith_product()
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            left = Binary(op, left, self._parse_arith_product())
        return left

    def _parse_arith_product(self) -> Expr:
        left = self._parse_arith_unary()
        while self.peek().type in (TokenType.STAR, TokenType.SLASH):
            op = self.advance().value
            left = Binary(op, left, self._parse_arith_unary())
        return left

    def _parse_arith_unary(self) -> Expr:
        if self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            operand = self._parse_arith_unary()
            if op == "+":
                return operand
            if isinstance(operand, Num):
                return Num(-operand.value)
            return Unary("-", operand)
        return self._parse_arith_power()

    def _parse_arith_power(self) -> Expr:
        base = self._parse_arith_atom()
        if self.peek().type in (TokenType.POWER, TokenType.CARET):
            self.advance()
            return Binary("**", base, self._parse_arith_unary())
        return base

    def _parse_arith_atom(self) -> Expr:
        token = self.peek()
        if token.type == TokenType.NUMBER:
            self.advance()
            return Num(float(token.value))
        if token.type == TokenType.STRING:
            self.advance()
            return Literal(token.value)
        if token.type == TokenType.LPAREN:
            self.advance()
            expr = self._parse_arith()
            self.expect(TokenType.RPAREN)
            return expr
        if self._at_call():
            self.advance()
            if token.value in _SPECIAL_FUNCTIONS:
                raise ValueError(
                    f"{token.value}() at position {token.position} cannot appear in an expression"
                )
            spec = TRANSFORM_FUNCTIONS.get(token.value)
            if spec is not None and not spec.nestable:
                raise ValueError(
                    f"{token.value}() at position {token.position} cannot be nested inside "
                    "another expression; use it as a formula term"
                )
            return self._parse_call(token.value, token.position, nested=True)
        if token.type == TokenType.IDENTIFIER:
            self.advance()
            return Var(token.value)
        self._unexpected("a number, variable, or function call")
        raise AssertionError("unreachable")

    def _parse_call_arguments(self, func: str, position: int) -> tuple[Expr, ...]:
        """Parse the data arguments of offset() and cbind()."""
        positional, keywords = self._parse_arguments()
        expected = 2 if func == "cbind" else 1
        if keywords or len(positional) != expected:
            raise ValueError(
                f"{func}() at position {position} takes {expected} unnamed argument"
                f"{'s' if expected > 1 else ''}"
            )
        for arg in positional:
            _require_data_expression(arg, func, position)
        return tuple(positional)

    def _parse_arguments(self) -> tuple[list[Expr], list[tuple[str, Expr]]]:
        self.expect(TokenType.LPAREN)
        positional: list[Expr] = []
        keywords: list[tuple[str, Expr]] = []
        if self.peek().type != TokenType.RPAREN:
            while True:
                if (
                    self.peek().type == TokenType.IDENTIFIER
                    and self.peek(1).type == TokenType.EQUALS
                ):
                    name = self.advance().value
                    self.advance()
                    keywords.append((name, self._parse_arith()))
                elif keywords:
                    self._unexpected("a named argument after named arguments")
                else:
                    positional.append(self._parse_arith())
                if self.peek().type != TokenType.COMMA:
                    break
                self.advance()
        self.expect(TokenType.RPAREN)
        return positional, keywords

    def _parse_call(self, func: str, position: int, *, nested: bool) -> Call:
        """Parse a function call; only a whole formula term must read a data column.

        Inside an expression, a constant call such as log(2) in I(x / log(2)) is allowed.
        """
        spec = TRANSFORM_FUNCTIONS.get(func)
        if spec is None:
            supported = ", ".join(sorted([*TRANSFORM_FUNCTIONS, *_SPECIAL_FUNCTIONS]))
            raise ValueError(
                f"Unknown formula function '{func}' at position {position}. "
                f"Supported functions: {supported}"
            )
        positional, keywords = self._parse_arguments()
        if len(positional) > len(spec.params):
            raise ValueError(
                f"{func}() at position {position} takes at most {len(spec.params)} arguments"
            )
        bound: dict[str, Expr] = {}
        for param, value in zip(spec.params, positional, strict=False):
            bound[param.name] = value
        names = [param.name for param in spec.params]
        for name, value in keywords:
            if name not in names:
                raise ValueError(f"{func}() at position {position} has no argument '{name}'")
            if name in bound:
                raise ValueError(f"{func}() at position {position} got '{name}' more than once")
            bound[name] = value

        data = bound.get("x")
        if data is None:
            raise ValueError(f"{func}() at position {position} requires a data argument")
        if nested:
            _reject_constant_literals(data, func, position)
        else:
            _require_data_expression(data, func, position)
        options: list[tuple[str, Expr]] = []
        for param in spec.params[1:]:
            if param.name not in bound:
                if param.default is _REQUIRED:
                    raise ValueError(
                        f"{func}() at position {position} requires argument '{param.name}'"
                    )
                if param.always:
                    options.append((param.name, _option_expr(param.default)))
                continue
            value = _bind_option(func, position, param, bound[param.name])
            if param.always or param.default is _REQUIRED or value != _option_expr(param.default):
                options.append((param.name, value))
        return Call(func, (data,), tuple(options))

    @staticmethod
    def _factor_term(factors: tuple[Factor, ...]) -> ParsedTerm:
        if len(factors) == 1:
            factor = factors[0]
            return VariableTerm(factor) if isinstance(factor, str) else factor
        return InteractionTerm(factors)

    @staticmethod
    def _term_variables(term: ParsedTerm) -> tuple[Factor, ...]:
        if isinstance(term, InterceptTerm):
            return ()
        if isinstance(term, VariableTerm):
            return (term.name,)
        if isinstance(term, PowerTerm | TransformTerm):
            return (term,)
        if isinstance(term, InteractionTerm):
            return term.variables
        assert_never(term)

    def _parse_random_term(self) -> tuple[list[RandomTerm], int | None]:
        """Parse a bar, with its position when it has no intercept or slopes."""
        start = self.expect(TokenType.LPAREN).position
        # lme4 builds each bar's columns from the bar's own formula.
        fixed_appearance, self._appearance = self._appearance, {}

        expr_terms: TermMap = {}
        has_intercept = True

        first = True
        while True:
            subtracting = False
            if self.peek().type in (TokenType.PLUS, TokenType.MINUS):
                subtracting = self.advance().type == TokenType.MINUS
            elif not first:
                if self.peek().type in (TokenType.PIPE, TokenType.DOUBLE_PIPE):
                    break
                self._unexpected("'+' or '-' between random-effect terms")
            if self.peek().type in (TokenType.NUMBER, TokenType.IDENTIFIER, TokenType.LPAREN):
                has_intercept = self._parse_effect(expr_terms, has_intercept, subtracting)
            else:
                self._unexpected("a random-effect term")
            first = False

        correlated = True
        if self.peek().type == TokenType.DOUBLE_PIPE:
            self.advance()
            correlated = False
        else:
            self.expect(TokenType.PIPE)

        groupings = self._parse_grouping()
        self.expect(TokenType.RPAREN)
        bar_terms = _in_variable_order(expr_terms.values(), self._appearance)
        self._appearance = fixed_appearance
        expr = _by_degree(bar_terms)
        parts = (
            [(expr, True, has_intercept)]
            if correlated
            else _split_uncorrelated(expr, has_intercept)
        )

        terms = [
            RandomTerm(
                expr=part,
                grouping=grouping,
                correlated=part_correlated,
                has_intercept=part_intercept,
            )
            for part, part_correlated, part_intercept in parts
            for grouping in groupings
        ]
        return terms, start if not expr and not has_intercept else None

    def _parse_grouping(self) -> list[str | tuple[str, ...]]:
        """Expand nesting to every prefix; colons identify a single joint factor."""
        groups: list[str] = []
        groupings: list[str | tuple[str, ...]] = []
        while True:
            group = self.expect(TokenType.IDENTIFIER).value
            if group not in groups:
                groups.append(group)
            while self.peek().type == TokenType.COLON:
                self.advance()
                group = self.expect(TokenType.IDENTIFIER).value
                if group not in groups:
                    groups.append(group)

            grouping = groups[0] if len(groups) == 1 else tuple(groups)
            if grouping not in groupings:
                groupings.append(grouping)
            if self.peek().type != TokenType.SLASH:
                return groupings
            self.advance()


def _as_power_term(expr: Expr) -> PowerTerm | None:
    """Recognize I(x**k) for a non-negative integer k, the historical power term."""
    if (
        isinstance(expr, Binary)
        and expr.op == "**"
        and isinstance(expr.left, Var)
        and isinstance(expr.right, Num)
        and expr.right.value >= 0
        and expr.right.value == int(expr.right.value)
    ):
        return PowerTerm(expr.left.name, int(expr.right.value))
    return None


def _require_data_expression(expr: Expr, func: str, position: int) -> None:
    _reject_constant_literals(expr, func, position)
    if not isinstance(expr, Var) and not _expr_has_variable(expr):
        raise ValueError(f"{func}() at position {position} must refer to a data variable")


def _reject_constant_literals(expr: Expr, func: str, position: int) -> None:
    def visit(node: Expr) -> None:
        if isinstance(node, Literal):
            raise ValueError(
                f"{func}() at position {position} cannot use the constant {format_expr(node)} "
                "as data"
            )
        if isinstance(node, Unary):
            visit(node.operand)
        elif isinstance(node, Binary):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, Call):
            for arg in node.args:
                visit(arg)

    visit(expr)


def _expr_has_variable(expr: Expr) -> bool:
    if isinstance(expr, Var):
        return True
    if isinstance(expr, Unary):
        return _expr_has_variable(expr.operand)
    if isinstance(expr, Binary):
        return _expr_has_variable(expr.left) or _expr_has_variable(expr.right)
    if isinstance(expr, Call):
        return any(_expr_has_variable(arg) for arg in expr.args)
    return False


def _option_expr(value: Any) -> Expr:
    if isinstance(value, bool | str):
        return Literal(value)
    if isinstance(value, int | float):
        return Num(float(value))
    if value is None:
        return Literal("")
    raise TypeError(f"Unsupported default {value!r}")


_CONSTANT_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "log1p": math.log1p,
    "exp": math.exp,
    "expm1": math.expm1,
    "sqrt": math.sqrt,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "floor": math.floor,
    "ceiling": math.ceil,
    "trunc": math.trunc,
    "sign": lambda value: float((value > 0) - (value < 0)),
    "I": float,
}


def _constant_value(expr: Expr) -> float | None:
    """Evaluate an expression without variables, or return None."""
    try:
        if isinstance(expr, Num):
            return expr.value
        if isinstance(expr, Unary):
            operand = _constant_value(expr.operand)
            return None if operand is None else -operand
        if isinstance(expr, Binary):
            left, right = _constant_value(expr.left), _constant_value(expr.right)
            if left is None or right is None:
                return None
            operations: dict[str, Callable[[float, float], float]] = {
                "+": lambda a, b: a + b,
                "-": lambda a, b: a - b,
                "*": lambda a, b: a * b,
                "/": lambda a, b: a / b,
                "**": lambda a, b: a**b,
            }
            return float(operations[expr.op](left, right))
        if isinstance(expr, Call) and expr.func in _CONSTANT_FUNCTIONS:
            argument = _constant_value(expr.args[0])
            if argument is None:
                return None
            if expr.func == "log" and expr.options:
                base = _constant_value(dict(expr.options)["base"])
                return None if base is None else math.log(argument, base)
            return float(_CONSTANT_FUNCTIONS[expr.func](argument))
    except (ArithmeticError, ValueError):
        return None
    return None


def _bind_option(func: str, position: int, param: TransformParam, value: Expr) -> Expr:
    """Validate a constant option and return its canonical expression."""
    where = f"{func}() at position {position}"
    if param.kind in ("integer", "number") and not isinstance(value, Num):
        # Options may be constant expressions, such as base = exp(1) or 1 + 1.
        constant = _constant_value(value)
        if constant is None:
            raise ValueError(f"{where} requires a constant number for '{param.name}'")
        value = Num(constant)
    if param.kind == "logical":
        if isinstance(value, Var) and value.name in _LOGICAL_NAMES:
            return Literal(_LOGICAL_NAMES[value.name])
        raise ValueError(f"{where} requires TRUE or FALSE for '{param.name}'")
    if param.kind == "integer":
        if isinstance(value, Num) and value.value == int(value.value) and value.value >= 1:
            return value
        raise ValueError(f"{where} requires a positive integer for '{param.name}'")
    if param.kind == "number":
        if isinstance(value, Num) and value.value > 0 and value.value != 1:
            return value
        raise ValueError(f"{where} requires a positive number other than 1 for '{param.name}'")
    if param.kind == "contrast":
        name = (
            value.name
            if isinstance(value, Var)
            else value.value
            if isinstance(value, Literal) and isinstance(value.value, str)
            else None
        )
        normalized = None if name is None else name.removeprefix("contr.").lower()
        if normalized == "diff":
            normalized = "deviation"
        if normalized not in _CONTRAST_NAMES:
            raise ValueError(f"{where} requires a contrast name: {', '.join(_CONTRAST_NAMES)}")
        return Literal(normalized)
    if param.kind == "level":
        if isinstance(value, Num) or (isinstance(value, Literal) and isinstance(value.value, str)):
            return value
        if isinstance(value, Var) and value.name in _LOGICAL_NAMES:
            return Literal(_LOGICAL_NAMES[value.name])
        raise ValueError(f"{where} requires a quoted level or a number for '{param.name}'")
    raise AssertionError(f"unknown option kind {param.kind}")


def parse_formula(formula: str) -> Formula:
    lexer = Lexer(formula)
    tokens = list(lexer.tokenize())
    parser = Parser(tokens)
    return replace(parser.parse(), source=formula)


def _bar_key(
    term: RandomTerm,
) -> tuple[str | tuple[str, ...], bool, bool, frozenset[frozenset[Factor]]]:
    """Identify a bar by its grouping and its set of slopes.

    The slopes form a set, so a bar matches however its slopes were ordered,
    including formulas saved before terms were ordered by degree. Covariance
    types have no formula spelling.
    """
    slopes = frozenset(frozenset(Parser._term_variables(slope)) for slope in term.expr)
    return term.grouping, term.correlated, term.has_intercept, slopes


_RESPONSE_FIELDS = ("response", "response_denominator", "response_expr", "response_failures")


def _response_of(formula: Formula) -> dict[str, Any]:
    return {name: getattr(formula, name) for name in _RESPONSE_FIELDS}


def _substitute_dot(expr: Expr, replacement: Expr) -> Expr:
    if isinstance(expr, Var):
        return replacement if expr.name == "." else expr
    if isinstance(expr, Unary):
        return Unary(expr.op, _substitute_dot(expr.operand, replacement))
    if isinstance(expr, Binary):
        return Binary(
            expr.op,
            _substitute_dot(expr.left, replacement),
            _substitute_dot(expr.right, replacement),
        )
    if isinstance(expr, Call):
        return replace(expr, args=tuple(_substitute_dot(arg, replacement) for arg in expr.args))
    return expr


def _substitute_response(new: Formula, old: Formula) -> Formula:
    """Replace ``.`` in a new response by the old response, as in ``log(.) ~ .``."""
    if old.grouped_response:
        raise ValueError(
            f"Cannot substitute the grouped response {old.response_expression} for '.'"
        )
    if new.response_denominator == ".":
        raise ValueError("'.' cannot be the trials of a successes / trials response")
    old_expr = old.response_expr if old.response_expr is not None else Var(old.response)
    successes = _substitute_dot(
        new.response_expr if new.response_expr is not None else Var(new.response), old_expr
    )
    plain = isinstance(successes, Var)
    return replace(
        new,
        response=successes.name if isinstance(successes, Var) else format_expr(successes),
        response_expr=None if plain else successes,
        response_failures=(
            None
            if new.response_failures is None
            else _substitute_dot(new.response_failures, old_expr)
        ),
    )


_POSITION = re.compile(r"position (\d+)")
_NUMBER = re.compile(r"(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _intercept_literal(term: str) -> bool | None:
    """Whether an update term ``1`` (True) or ``0`` (False) names the intercept.

    Numbers compare by value, as in the formula grammar, so ``1.`` and ``0.0`` count.
    """
    if _NUMBER.fullmatch(term):
        value = float(term)
        if value in (0.0, 1.0):
            return value == 1.0
    return None


def _parse_piece(
    text: str, prefix: str, suffix: str, offset: int, end: int | None = None
) -> Formula:
    """Parse ``prefix + text + suffix``, reporting positions within the caller's string.

    ``offset`` is where ``text`` starts in the string the caller passed, and
    ``end``, when given, is where positions past ``text`` are reported.
    """
    try:
        return parse_formula(f"{prefix}{text}{suffix}")
    except ValueError as error:
        shift = offset - len(prefix)
        past = len(prefix) + len(text)

        def position(match: re.Match[str]) -> str:
            value = int(match[1])
            return f"position {end if end is not None and value >= past else value + shift}"

        raise ValueError(_POSITION.sub(position, str(error))) from None


def update_formula(old_formula: Formula, new_formula_str: str) -> Formula:
    """Update a formula as R's update() does.

    Each '.' on the right-hand side stands for the old right-hand side: the old
    fixed terms wherever it appears, as in ``. ~ k + .``, ``. ~ .^2`` or
    ``. ~ . * x``, and the old random terms and offsets at the formula term that
    holds it, with the term's sign, so ``- .`` and ``- (.)`` remove them. As R's
    update() simplifies the result, terms are then ordered by degree and each
    interaction lists its factors in their order of first appearance.
    """
    lead = len(new_formula_str) - len(new_formula_str.lstrip())
    new_formula_str = new_formula_str.strip()

    lhs, rhs = _split_formula_parts(new_formula_str)
    lhs_offset = lead + len(lhs) - len(lhs.lstrip())
    rhs_offset = lead + len(new_formula_str) - len(rhs.lstrip())
    tilde = lead + len(lhs)
    # A one-sided update keeps the old response, as in R.
    lhs = lhs.strip() or "."
    rhs = rhs.strip()

    response_source = (
        old_formula if lhs == "." else _parse_piece(lhs, "", " ~ 1", lhs_offset, end=tilde)
    )
    if response_source is not old_formula and "." in response_source.response_variables:
        response_source = _substitute_response(response_source, old_formula)
    response = _response_of(response_source)

    if rhs == ".":
        return _simplified(_distinct_bars(replace(old_formula, **response), old_formula))

    prefix = f"{response_source.response_expression} ~ "

    def located(error: ValueError, translate: Callable[[int], int]) -> ValueError:
        message = _POSITION.sub(
            lambda match: f"position {rhs_offset + translate(int(match[1]) - len(prefix))}",
            str(error),
        )
        return ValueError(message)

    try:
        # Read the update as written first, each '.' as a name, so syntax errors
        # name its own tokens rather than the text that replaces them.
        Parser(list(Lexer(prefix + rhs).tokenize()), dot_groups=True).parse()
    except ValueError as error:
        raise located(error, lambda position: position) from None
    try:
        text, translate, empty = _substitute_dots(rhs, old_formula)
    except ValueError as error:
        # The lexer reports positions within the right-hand side.
        raise located(error, lambda position: position + len(prefix)) from None
    try:
        # An old fixed part with no terms adds none and leaves none in products.
        parsed = Parser(list(Lexer(prefix + text).tokenize()), none=empty).parse()
    except ValueError as error:
        raise located(error, translate) from None
    # As R reads the substituted formula, the old '- 1' takes effect at each '.'.
    parsed = replace(
        parsed,
        fixed=replace(parsed.fixed, has_intercept=_updated_intercept(rhs, old_formula)),
    )
    updated = replace(old_formula, fixed=parsed.fixed, random=parsed.random, offsets=parsed.offsets)
    return _simplified(_distinct_bars(replace(updated, **response), old_formula))


def _distinct_bars(formula: Formula, old_formula: Formula) -> Formula:
    """Keep each distinct bar once, at its first place, as R's simplification does,
    and the old random terms themselves, with their covariance types."""
    old_bars = {_bar_key(term): term for term in old_formula.random}
    distinct: dict[Any, RandomTerm] = {}
    for term in formula.random:
        key = _bar_key(term)
        distinct.setdefault(key, old_bars.get(key, term))
    return replace(formula, random=tuple(distinct.values()))


def _updated_intercept(rhs: str, old_formula: Formula) -> bool:
    """Whether an update keeps the intercept, reading 0, 1 and each '.' in order.

    As R reads the substituted formula, each '.' brings the old right-hand side
    with its intercept removal, which a subtracted term reverses; a '.' inside a
    function call or a random-effect bar leaves the fixed intercept alone, and a
    later 0 or 1 term overrides what came before.
    """
    tokens = list(Lexer(rhs).tokenize())
    # The intercept the old formula writes, such as its 0 or an explicit 1.
    _, _, old_intercept, _ = _written_fixed(old_formula)
    has_intercept = True
    # One frame per open parenthesis: whether it is a function call or a bar, and
    # whether the term it is in so far is subtracted; the outer frame is the formula.
    frames: list[list[bool]] = [[False, False, False]]
    for index, token in enumerate(tokens):
        before = tokens[index - 1].type if index else None
        frame = frames[-1]
        if token.type in (TokenType.PLUS, TokenType.MINUS):
            frame[2] = token.type == TokenType.MINUS
        elif token.type == TokenType.LPAREN:
            call = before == TokenType.IDENTIFIER
            frames.append([call, not call and _starts_bar(tokens, index), False])
        elif token.type == TokenType.RPAREN and len(frames) > 1:
            frames.pop()
        elif (
            token.type == TokenType.NUMBER
            and len(frames) == 1
            and before in (None, TokenType.PLUS, TokenType.MINUS)
            and tokens[index + 1].type in (TokenType.EOF, TokenType.PLUS, TokenType.MINUS)
            and float(token.value) in (0.0, 1.0)
        ):
            has_intercept = (not frame[2]) == (float(token.value) == 1.0)
        elif (
            token.type == TokenType.IDENTIFIER
            and token.value == "."
            and old_intercept is not None
            and not any(call or bar for call, bar, _ in frames)
        ):
            # Each enclosing subtraction reverses the old intercept term, as R reads it.
            subtracted = sum(minus for _, _, minus in frames) % 2 == 1
            has_intercept = old_intercept != subtracted
    return has_intercept


def _bar_term_end(tokens: list[Token], index: int, depth: int) -> int:
    """Where the bar-level term holding the token at ``index`` ends.

    ``depth`` counts the parentheses between the bar and the token.
    """
    for token in tokens[index + 1 :]:
        if token.type == TokenType.LPAREN:
            depth += 1
        elif token.type == TokenType.RPAREN:
            depth -= 1
            if depth == 0:
                return token.position
        elif depth == 1 and token.type in (
            TokenType.PLUS,
            TokenType.MINUS,
            TokenType.PIPE,
            TokenType.DOUBLE_PIPE,
        ):
            return token.position
    return tokens[-1].position


def _starts_bar(tokens: list[Token], index: int) -> bool:
    """Whether the parenthesis at ``index`` opens a random-effect bar."""
    depth = 0
    for token in tokens[index:]:
        if token.type == TokenType.LPAREN:
            depth += 1
        elif token.type == TokenType.RPAREN:
            depth -= 1
            if depth == 0:
                return False
        elif token.type in (TokenType.PIPE, TokenType.DOUBLE_PIPE) and depth == 1:
            return True
    return False


def _simplified(formula: Formula) -> Formula:
    """Order terms by degree and list each interaction's factors in their order of
    first appearance in the result, as R's update() simplifies its formula."""
    terms = _by_degree(formula.fixed.terms)
    order: dict[Factor, int] = {}
    for term in terms:
        for factor in term_factors(term):
            order.setdefault(factor, len(order))
    terms = _by_degree(_in_variable_order(terms, order))
    # The result is R's simplified formula, which its terms now describe.
    return replace(formula, fixed=replace(formula.fixed, terms=terms), source=None)


def _written_fixed(formula: Formula) -> tuple[str | None, str | None, bool | None, bool]:
    """The fixed part of a formula as written, and its explicit intercept term.

    R's update() substitutes the old formula as written for '.', so the text it
    was parsed from is used while it still describes the formula. Returns the
    fixed terms' text without and with its 0 and 1 terms (None when the source
    is not available), the intercept a written 0 or 1 sets (None when none is
    written), and whether the source was used.
    """
    source = getattr(formula, "source", None)
    if source is not None:
        try:
            reparsed = parse_formula(source)
        except ValueError:
            reparsed = None
        # Only the fixed part of the source is used, so it is valid while the
        # fixed part is unchanged, as after set_cov_type() or nobars().
        if reparsed is not None and reparsed.fixed == formula.fixed:
            _, rhs = _split_formula_parts(source)
            tokens = list(Lexer(rhs).tokenize())
            spans: list[tuple[str, int, int, int]] = []
            sign, start, first, depth = "+", -1, -1, 0
            for index, token in enumerate(tokens):
                if token.type == TokenType.EOF:
                    break
                if depth == 0 and token.type in (TokenType.PLUS, TokenType.MINUS):
                    if start >= 0:
                        spans.append((sign, start, token.position, first))
                    sign, start = ("-" if token.type == TokenType.MINUS else "+"), -1
                    continue
                if start < 0:
                    start, first = token.position, index
                if token.type == TokenType.LPAREN:
                    depth += 1
                elif token.type == TokenType.RPAREN:
                    depth -= 1
            if start >= 0:
                spans.append((sign, start, len(rhs), first))
            kept: list[str] = []
            raw: list[str] = []
            intercept: bool | None = None
            for sign, start, end, first in spans:
                text = rhs[start:end].strip()
                token = tokens[first]
                if token.type == TokenType.NUMBER and text == token.value:
                    if float(token.value) in (0.0, 1.0):
                        intercept = (sign == "+") == (float(token.value) == 1.0)
                    raw.append(f"{sign} {text}")
                    continue
                if token.type == TokenType.LPAREN and _starts_bar(tokens, first):
                    continue
                if token.type == TokenType.IDENTIFIER and token.value == "offset":
                    continue
                kept.append(f"{sign} {text}")
                raw.append(f"{sign} {text}")
            written = " ".join(kept).removeprefix("+ ")
            return written, " ".join(raw).removeprefix("+ "), intercept, True
    return None, None, None if formula.fixed.has_intercept else False, False


def _substitute_dots(
    rhs: str, old_formula: Formula
) -> tuple[str, Callable[[int], int], str | None]:
    """Replace each '.' of an update's right-hand side by the old right-hand side.

    The old intercept term also enters bars that hold a '.', and the old random
    terms and offsets enter at the formula terms that hold one. Returns the new
    text, a map from its positions back to positions in ``rhs``, and the name
    standing for an old fixed part with no terms, which the parser reads as none.
    """
    from mixedlm.formula.terms import _format_printed, _printing_order

    tokens = list(Lexer(rhs).tokenize())
    written, raw, intercept, _ = _written_fixed(old_formula)
    fixed = (
        written
        if written is not None
        else " + ".join(_format_printed(item) for item in _printing_order(old_formula.fixed.terms))
    )
    extras = [_format_random(term) for term in old_formula.random]
    extras += [f"offset({format_expr(offset)})" for offset in old_formula.offsets]
    names = {token.value for token in tokens if token.type == TokenType.IDENTIFIER}
    empty = None
    # Edits as (position in rhs, characters replaced, text); each '.' is replaced,
    # and the old intercept term and random terms are inserted.
    edits: list[tuple[int, int, str]] = []
    # The formula terms holding a '.' that is not a separate term, by their first
    # token, with those dots, and where each formula term ends.
    holders: dict[int, list[int]] = {}
    ends: dict[int, int] = {}
    # One frame per open parenthesis: [call, bar, subtracted, past a bar's '|'];
    # the first is the formula.
    frames: list[list[bool]] = [[False, False, False, False]]
    term_start = 0
    for index, token in enumerate(tokens):
        frame = frames[-1]
        before = tokens[index - 1].type if index else None
        if token.type == TokenType.EOF or (
            len(frames) == 1 and token.type in (TokenType.PLUS, TokenType.MINUS)
        ):
            ends[term_start] = token.position
            term_start = index + 1
        if token.type in (TokenType.PLUS, TokenType.MINUS):
            frame[2] = token.type == TokenType.MINUS
            continue
        if token.type in (TokenType.PIPE, TokenType.DOUBLE_PIPE):
            frame[3] = True
            continue
        if token.type == TokenType.LPAREN:
            call = before == TokenType.IDENTIFIER
            frames.append([call, not call and _starts_bar(tokens, index), False, False])
            continue
        if token.type == TokenType.RPAREN and len(frames) > 1:
            frames.pop()
            continue
        if token.type != TokenType.IDENTIFIER or token.value != ".":
            continue
        after = tokens[index + 1].type
        bars = [i for i, frame in enumerate(frames) if frame[1]]
        in_call = any(frame[0] for frame in frames)
        if bars and frames[bars[-1]][3]:
            raise ValueError(
                f"'.' at position {token.position} stands for the old fixed terms, "
                "which cannot group random effects"
            )
        standalone = (
            len(frames) == 1
            and before in (None, TokenType.PLUS, TokenType.MINUS)
            and after in (TokenType.EOF, TokenType.PLUS, TokenType.MINUS)
        )
        if standalone:
            sign = "-" if before == TokenType.MINUS else "+"
            parts = ([f"({fixed})"] if fixed else []) + extras
            # An empty old right-hand side leaves only its intercept, which
            # _updated_intercept settles.
            replacement = f" {sign} ".join(parts) or "1"
        elif in_call:
            if not fixed:
                raise ValueError(
                    f"'.' at position {token.position} stands for the old fixed terms, "
                    "and the old formula has none"
                )
            # Inside a call the old right-hand side is an argument, as written.
            replacement = f"({raw or fixed})"
        elif fixed:
            replacement = f"({fixed})"
        else:
            # With no old fixed terms the '.' adds none, in the fixed part or a bar,
            # and a product with it has none; a name the parser reads so stands for it.
            if empty is None:
                empty = "._none"
                while empty in names:
                    empty += "_"
            replacement = f"({empty})"
        edits.append((token.position, _dot_width(rhs, token.position), replacement))
        if not standalone:
            holders.setdefault(term_start, []).append(index)
        if bars and not in_call and intercept is not None:
            # In a bar the old 0 or 1 applies too, reversed by each subtraction
            # between the bar and the '.', as R reads the substituted formula.
            subtracted = sum(frame[2] for frame in frames[bars[-1] :]) % 2 == 1
            value = "1" if intercept != subtracted else "0"
            position = _bar_term_end(tokens, index, len(frames) - bars[-1])
            edits.append((position, 0, f" + {value}"))
    if extras:
        for first, dots in holders.items():
            joined = _random_terms_sign(rhs, tokens, first, ends[first], dots)
            if joined is not None:
                # The old random terms and offsets take the place of the '.', as
                # R reads the substituted formula, so later terms can remove them.
                text = "".join(f" {joined} {extra}" for extra in extras)
                edits.append((ends[first], 0, text))
    edits.sort(key=lambda edit: (edit[0], -edit[1]))
    return (*_apply_edits(rhs, edits), empty)


def _dot_width(rhs: str, position: int) -> int:
    """The length of the '.' token at ``position``; a backticked `.` is the same name."""
    return _group_end(rhs, position) - position if rhs[position] == "`" else 1


def _random_terms_sign(
    rhs: str, tokens: list[Token], first: int, end: int, dots: list[int]
) -> str | None:
    """How the old random terms enter at a formula term that holds a '.'.

    R expands the substituted old right-hand side with the term, so its random
    terms are added when they remain anywhere in the term, as in ``(.):w`` or
    ``log(.)``, and removed by a subtracted term only when they remain as
    separate terms, as in ``- (. + w)``; a subtraction inside the term, as in
    ``(z - .)``, removes them from it. Returns '+', '-' or None.
    """
    start = tokens[first].position
    names = {token.value for token in tokens if token.type == TokenType.IDENTIFIER}
    sentinel = "._dot"
    while sentinel in names:
        sentinel += "_"
    pieces, cursor = [], start
    for index in dots:
        position = tokens[index].position
        pieces += [rhs[cursor:position], f"({sentinel})"]
        cursor = position + _dot_width(rhs, position)
    pieces.append(rhs[cursor:end])
    subtracted = first > 0 and tokens[first - 1].type == TokenType.MINUS
    try:
        term = parse_formula(f"{sentinel}_ ~ {''.join(pieces)}")
    except ValueError:
        # The complete update reports the error.
        return None if subtracted else "+"
    if subtracted:
        return "-" if VariableTerm(sentinel) in term.fixed.terms else None
    variables = (
        term.fixed_variables | term.random_variables | term.grouping_factors | term.offset_variables
    )
    return "+" if sentinel in variables else None


def _apply_edits(rhs: str, edits: list[tuple[int, int, str]]) -> tuple[str, Callable[[int], int]]:
    """Apply sorted edits to ``rhs``, with a map from new positions back to ``rhs``."""
    pieces: list[str] = []
    regions: list[tuple[int, int, int, int]] = []
    cursor = length = 0
    for position, removed, text in edits:
        pieces.append(rhs[cursor:position])
        length += position - cursor
        regions.append((length, length + len(text), position, removed))
        pieces.append(text)
        length += len(text)
        cursor = position + removed
    pieces.append(rhs[cursor:])

    def translate(position: int) -> int:
        shift = 0
        for new_start, new_end, original, removed in regions:
            if position < new_start:
                break
            if position < new_end:
                return original
            shift = new_end - (original + removed)
        return position - shift

    return "".join(pieces), translate


def _split_formula_parts(formula: str) -> tuple[str, str]:
    i = 0
    while i < len(formula):
        if formula[i] == "`":
            i = _group_end(formula, i)
        elif formula[i] == "~":
            return formula[:i], formula[i + 1 :]
        else:
            i += 1
    raise ValueError("Formula must contain '~'")


def _group_end(text: str, start: int) -> int:
    """Find the end of a parenthesized expression or quoted name in one pass."""
    depth = 0
    quote = ""
    i = start
    while i < len(text):
        char = text[i]
        i += 1
        if quote:
            if char == "\\":
                i += 1
            elif char == quote:
                quote = ""
        elif char in ("`", "'", '"'):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth == 0 and not quote:
            return i
    raise ValueError("Unterminated quoted identifier or parenthesis in formula")


def _additive_term_offsets(rhs: str) -> list[int]:
    """Where each term of :func:`_split_additive_terms` starts in ``rhs``."""
    offsets: list[int] = []
    start = i = 0
    while i < len(rhs):
        char = rhs[i]
        if char in ("(", "`", "'", '"'):
            i = _group_end(rhs, i)
            continue
        if char in "+-":
            if rhs[start:i].strip():
                offsets.append(start + len(rhs[start:i]) - len(rhs[start:i].lstrip()))
            start = i + 1
        i += 1
    offsets.append(start + len(rhs[start:]) - len(rhs[start:].lstrip()))
    return offsets


def _split_additive_terms(rhs: str) -> list[tuple[str, str]]:
    updates: list[tuple[str, str]] = []
    operation = "+"
    start = i = 0
    while i < len(rhs):
        char = rhs[i]
        if char in ("(", "`", "'", '"'):
            i = _group_end(rhs, i)
            continue
        if char == ")":
            raise ValueError(f"Unexpected ')' at position {i}")
        if char in "+-":
            term = rhs[start:i].strip()
            if term:
                updates.append((operation, term))
            elif start:
                raise ValueError("Expected a formula update term between '+' and '-' operators")
            operation = char
            start = i + 1
        i += 1

    term = rhs[start:].strip()
    if not term:
        raise ValueError("Expected a formula update term")
    updates.append((operation, term))
    return updates


def nobars(formula: Formula | str) -> Formula:
    """Remove random effects (bar terms) from a formula.

    Returns a new formula containing only the fixed effects part,
    with all random effects removed.

    Parameters
    ----------
    formula : Formula or str
        A Formula object or formula string to process.

    Returns
    -------
    Formula
        A new formula with random effects removed.

    Examples
    --------
    >>> f = parse_formula("y ~ x + (1 | group)")
    >>> nobars(f)
    Formula(response='y', fixed=..., random=())

    >>> nobars("y ~ x + (x | group) + (1 | subject)")
    Formula(response='y', fixed=..., random=())
    """
    if isinstance(formula, str):
        formula = parse_formula(formula)

    return replace(formula, random=())


def findbars(formula: Formula | str) -> tuple[RandomTerm, ...]:
    """Find and return the random effects (bar terms) from a formula.

    Extracts all random effect specifications from a mixed model formula.

    Parameters
    ----------
    formula : Formula or str
        A Formula object or formula string to process.

    Returns
    -------
    tuple of RandomTerm
        The random effect terms found in the formula.

    Examples
    --------
    >>> f = parse_formula("y ~ x + (1 | group)")
    >>> bars = findbars(f)
    >>> len(bars)
    1
    >>> bars[0].grouping
    'group'

    >>> bars = findbars("y ~ x + (x | group) + (1 | subject)")
    >>> len(bars)
    2
    """
    if isinstance(formula, str):
        formula = parse_formula(formula)

    return formula.random


def subbars(formula: Formula | str) -> str:
    """Substitute random effects with fixed effects equivalents.

    Converts random effect terms to their fixed effect equivalents.
    For example, `(1 + x | group)` becomes `group + group:x`.

    Parameters
    ----------
    formula : Formula or str
        A Formula object or formula string to process.

    Returns
    -------
    str
        A formula string with random effects converted to fixed effects.

    Examples
    --------
    >>> subbars("y ~ x + (1 | group)")
    'y ~ x + group'

    >>> subbars("y ~ x + (x | group)")
    'y ~ x + group + group:x'
    """
    if isinstance(formula, str):
        formula = parse_formula(formula)

    fixed_parts: list[str] = []

    if not formula.fixed.has_intercept:
        fixed_parts.append("0")

    from mixedlm.formula.terms import _format_printed, _printing_order

    # Print in an order that keeps each interaction's factor order on parsing.
    for item in _printing_order(formula.fixed.terms):
        if isinstance(item, InterceptTerm):
            continue
        fixed_parts.append(_format_printed(item))

    fixed_parts.extend(f"offset({format_expr(offset)})" for offset in formula.offsets)

    for rterm in formula.random:
        grouping = _format_grouping(rterm.grouping)

        if rterm.has_intercept:
            fixed_parts.append(grouping)

        for term in rterm.expr:
            if isinstance(term, InterceptTerm):
                continue
            fixed_parts.append(f"{grouping}:{_format_term(term)}")

    rhs = " + ".join(fixed_parts) if fixed_parts else "1"
    return f"{formula.response_expression} ~ {rhs}"


def is_mixed_formula(formula: Formula | str) -> bool:
    """Check if a formula contains random effects.

    Parameters
    ----------
    formula : Formula or str
        A Formula object or formula string to check.

    Returns
    -------
    bool
        True if the formula contains random effects, False otherwise.

    Examples
    --------
    >>> is_mixed_formula("y ~ x + (1 | group)")
    True

    >>> is_mixed_formula("y ~ x")
    False
    """
    if isinstance(formula, str):
        formula = parse_formula(formula)

    return len(formula.random) > 0


def set_cov_type(
    formula: Formula | str,
    cov_type: str | dict[str, str],
) -> Formula:
    """Set the covariance structure type for random effects.

    Parameters
    ----------
    formula : Formula or str
        A Formula object or formula string to modify.
    cov_type : str or dict
        The covariance structure type. Can be:
        - "us": Unstructured (default) - full covariance matrix
        - "cs": Compound symmetry - equal correlations
        - "ar1": Autoregressive order 1 - correlation decays with distance
        If a dict, keys are grouping factor names and values are cov types.

    Returns
    -------
    Formula
        A new formula with the specified covariance structure.

    Examples
    --------
    >>> f = parse_formula("y ~ x + (time | subject)")
    >>> f_cs = set_cov_type(f, "cs")

    >>> f_mixed = set_cov_type(f, {"subject": "ar1"})
    """
    if isinstance(formula, str):
        formula = parse_formula(formula)

    valid_types = {"us", "cs", "ar1"}

    if isinstance(cov_type, str):
        if cov_type not in valid_types:
            raise ValueError(f"Invalid cov_type '{cov_type}'. Must be one of {valid_types}")
        cov_map = {_format_grouping(rt.grouping): cov_type for rt in formula.random}
    else:
        for ct in cov_type.values():
            if ct not in valid_types:
                raise ValueError(f"Invalid cov_type '{ct}'. Must be one of {valid_types}")
        cov_map = cov_type

    new_random = []
    for rt in formula.random:
        group_key = _format_grouping(rt.grouping)
        new_cov = cov_map.get(group_key, rt.cov_type)
        new_rt = RandomTerm(
            expr=rt.expr,
            grouping=rt.grouping,
            correlated=rt.correlated,
            has_intercept=rt.has_intercept,
            cov_type=new_cov,
        )
        new_random.append(new_rt)

    return replace(formula, random=tuple(new_random))


def expandDoubleVerts(formula: str) -> str:
    """Expand || notation to explicit uncorrelated random effects.

    The || notation in lme4 specifies uncorrelated random effects.
    For example, (1 + x || group) is equivalent to (1 | group) + (0 + x | group).
    This expansion preserves the fitted model for single-column slopes such
    as numeric predictors. A categorical slope can encode multiple columns.
    Fit the original || formula to keep those columns independent.

    Parameters
    ----------
    formula : str
        A formula string that may contain || notation.

    Returns
    -------
    str
        The formula string with || expanded to uncorrelated terms.

    Examples
    --------
    >>> expandDoubleVerts("y ~ x + (1 + x || group)")
    'y ~ x + (1 | group) + (0 + x | group)'

    >>> expandDoubleVerts("y ~ x + (time || subject)")
    'y ~ x + (1 | subject) + (0 + time | subject)'
    """
    parsed = parse_formula(formula)
    if all(term.correlated for term in parsed.random):
        return formula

    expanded: list[RandomTerm] = []
    for random_term in parsed.random:
        if random_term.correlated:
            expanded.append(random_term)
            continue

        if random_term.has_intercept:
            expanded.append(
                RandomTerm(
                    expr=(),
                    grouping=random_term.grouping,
                    correlated=True,
                    has_intercept=True,
                    cov_type=random_term.cov_type,
                )
            )

        expanded.extend(
            RandomTerm(
                expr=(term,),
                grouping=random_term.grouping,
                correlated=True,
                has_intercept=False,
                cov_type=random_term.cov_type,
            )
            for term in random_term.expr
            if not isinstance(term, InterceptTerm)
        )

    return str(replace(parsed, random=tuple(expanded)))


def dropOffset(formula: str | Formula) -> str:
    """Remove offset terms from a formula.

    Removes any offset() terms from the formula string.

    Parameters
    ----------
    formula : str or Formula
        A formula string or Formula object that may contain offset terms.

    Returns
    -------
    str
        The formula string with offset terms removed.

    Examples
    --------
    >>> dropOffset("y ~ x + offset(log(t)) + (1 | group)")
    'y ~ x + (1 | group)'

    >>> dropOffset("y ~ x + offset(exposure)")
    'y ~ x'
    """
    formula_str = str(formula)
    if "offset" not in formula_str:
        return formula_str
    lhs, rhs = _split_formula_parts(formula_str)
    terms = _split_additive_terms(rhs)
    kept = []
    for sign, term in terms:
        if term.startswith("offset"):
            args = term[len("offset") :].lstrip()
            if args.startswith("(") and _group_end(args, 0) == len(args):
                continue
        kept.append((sign, term))

    if len(kept) == len(terms):
        return formula_str
    pieces = [
        term if i == 0 and sign == "+" else f"{sign} {term}" for i, (sign, term) in enumerate(kept)
    ]
    return f"{lhs.strip()} ~ {' '.join(pieces) or '1'}"


def getResponseName(formula: Formula | str) -> str:
    """Get the name of the response variable from a formula.

    Parameters
    ----------
    formula : Formula or str
        A Formula object or formula string.

    Returns
    -------
    str
        The name of the response variable.

    Examples
    --------
    >>> getResponseName("y ~ x + (1 | group)")
    'y'

    >>> getResponseName("Reaction ~ Days + (Days | Subject)")
    'Reaction'
    """
    if isinstance(formula, str):
        formula = parse_formula(formula)

    return formula.response


def getFixedFormulaStr(formula: Formula | str) -> str:
    """Get the fixed effects part of a formula as a string.

    Parameters
    ----------
    formula : Formula or str
        A Formula object or formula string.

    Returns
    -------
    str
        The fixed effects formula string (without random effects).

    Examples
    --------
    >>> getFixedFormulaStr("y ~ x + z + (1 | group)")
    'y ~ x + z'

    >>> getFixedFormulaStr("y ~ x * z + (x | group)")
    'y ~ x + z + x:z'
    """
    if isinstance(formula, str):
        formula = parse_formula(formula)

    return str(replace(formula, random=()))


def getRandomFormulaStr(formula: Formula | str) -> str:
    """Get the random effects part of a formula as a string.

    Parameters
    ----------
    formula : Formula or str
        A Formula object or formula string.

    Returns
    -------
    str
        The random effects formula string.

    Examples
    --------
    >>> getRandomFormulaStr("y ~ x + (1 | group)")
    '(1 | group)'

    >>> getRandomFormulaStr("y ~ x + (x | group) + (1 | subject)")
    '(x | group) + (1 | subject)'
    """
    if isinstance(formula, str):
        formula = parse_formula(formula)

    return " + ".join(_format_random(term) for term in formula.random)


def getNGroups(formula: Formula | str) -> int:
    """Count the number of grouping factors in a formula.

    Parameters
    ----------
    formula : Formula or str
        A Formula object or formula string.

    Returns
    -------
    int
        The number of distinct grouping factors.

    Examples
    --------
    >>> getNGroups("y ~ x + (1 | group)")
    1

    >>> getNGroups("y ~ x + (1 | group) + (1 | subject)")
    2
    """
    if isinstance(formula, str):
        formula = parse_formula(formula)

    groupings: set[str | tuple[str, ...]] = set()
    for rterm in formula.random:
        if isinstance(rterm.grouping, tuple):
            groupings.add(tuple(rterm.grouping))
        else:
            groupings.add(rterm.grouping)

    return len(groupings)
