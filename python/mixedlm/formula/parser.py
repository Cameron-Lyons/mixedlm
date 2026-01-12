from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto

from mixedlm.formula.terms import (
    FixedTerm,
    Formula,
    InteractionTerm,
    InterceptTerm,
    RandomTerm,
    VariableTerm,
)


class TokenType(Enum):
    TILDE = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    COLON = auto()
    SLASH = auto()
    PIPE = auto()
    DOUBLE_PIPE = auto()
    LPAREN = auto()
    RPAREN = auto()
    NUMBER = auto()
    IDENTIFIER = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    position: int


class Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def peek(self) -> str | None:
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def advance(self) -> str | None:
        ch = self.peek()
        self.pos += 1
        return ch

    def skip_whitespace(self) -> None:
        while self.peek() is not None and self.peek() in " \t\n\r":
            self.advance()

    def tokenize(self) -> Iterator[Token]:
        while True:
            self.skip_whitespace()
            start_pos = self.pos
            ch = self.peek()

            if ch is None:
                yield Token(TokenType.EOF, "", start_pos)
                return

            if ch == "~":
                self.advance()
                yield Token(TokenType.TILDE, "~", start_pos)
            elif ch == "+":
                self.advance()
                yield Token(TokenType.PLUS, "+", start_pos)
            elif ch == "-":
                self.advance()
                yield Token(TokenType.MINUS, "-", start_pos)
            elif ch == "*":
                self.advance()
                yield Token(TokenType.STAR, "*", start_pos)
            elif ch == ":":
                self.advance()
                yield Token(TokenType.COLON, ":", start_pos)
            elif ch == "/":
                self.advance()
                yield Token(TokenType.SLASH, "/", start_pos)
            elif ch == "|":
                self.advance()
                if self.peek() == "|":
                    self.advance()
                    yield Token(TokenType.DOUBLE_PIPE, "||", start_pos)
                else:
                    yield Token(TokenType.PIPE, "|", start_pos)
            elif ch == "(":
                self.advance()
                yield Token(TokenType.LPAREN, "(", start_pos)
            elif ch == ")":
                self.advance()
                yield Token(TokenType.RPAREN, ")", start_pos)
            elif ch.isdigit():
                num = ""
                while self.peek() is not None and self.peek().isdigit():
                    num += self.advance()  # type: ignore[operator]
                yield Token(TokenType.NUMBER, num, start_pos)
            elif ch.isalpha() or ch == "_" or ch == ".":
                ident = ""
                while (
                    self.peek() is not None and (self.peek().isalnum() or self.peek() in "_.")  # type: ignore[union-attr]
                ):
                    ident += self.advance()  # type: ignore[operator]
                yield Token(TokenType.IDENTIFIER, ident, start_pos)
            else:
                raise ValueError(f"Unexpected character '{ch}' at position {start_pos}")


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def peek(self) -> Token:
        return self.current()

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

    def parse(self) -> Formula:
        response = self._parse_response()
        self.expect(TokenType.TILDE)
        fixed_terms, random_terms = self._parse_rhs()

        has_intercept = True
        filtered_terms: list[InterceptTerm | VariableTerm | InteractionTerm] = []
        for term in fixed_terms:
            if isinstance(term, tuple) and term[0] == "no_intercept":
                has_intercept = False
            elif isinstance(term, InterceptTerm | VariableTerm | InteractionTerm):
                filtered_terms.append(term)

        fixed = FixedTerm(terms=tuple(filtered_terms), has_intercept=has_intercept)
        return Formula(response=response, fixed=fixed, random=tuple(random_terms))

    def _parse_response(self) -> str:
        tok = self.expect(TokenType.IDENTIFIER)
        return tok.value

    def _parse_rhs(
        self,
    ) -> tuple[
        list[InterceptTerm | VariableTerm | InteractionTerm | tuple[str, ...]],
        list[RandomTerm],
    ]:
        fixed_terms: list[InterceptTerm | VariableTerm | InteractionTerm | tuple[str, ...]] = []
        random_terms: list[RandomTerm] = []

        while self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.LPAREN:
                random_terms.append(self._parse_random_term())
            elif self.peek().type in (TokenType.IDENTIFIER, TokenType.NUMBER):
                term = self._parse_term()
                if term is not None:
                    fixed_terms.append(term)
            elif self.peek().type == TokenType.MINUS:
                self.advance()
                next_term = self._parse_term()
                if isinstance(next_term, InterceptTerm):
                    fixed_terms.append(("no_intercept",))
            elif self.peek().type == TokenType.PLUS:
                self.advance()
            else:
                break

        return fixed_terms, random_terms

    def _parse_term(
        self,
    ) -> InterceptTerm | VariableTerm | InteractionTerm | tuple[str, ...] | None:
        if self.peek().type == TokenType.NUMBER:
            tok = self.advance()
            if tok.value == "1":
                return InterceptTerm()
            elif tok.value == "0":
                return ("no_intercept",)
            else:
                raise ValueError(f"Unexpected number {tok.value} in formula")

        base = self._parse_base_term()
        if base is None:
            return None

        while self.peek().type in (TokenType.COLON, TokenType.STAR):
            op = self.advance()
            next_base = self._parse_base_term()
            if next_base is None:
                continue

            if op.type == TokenType.COLON:
                base = self._combine_interaction(base, next_base)
            else:
                base = self._combine_star(base, next_base)

        return base

    def _parse_base_term(self) -> VariableTerm | InteractionTerm | None:
        if self.peek().type != TokenType.IDENTIFIER:
            return None
        tok = self.advance()
        return VariableTerm(tok.value)

    def _combine_interaction(
        self,
        left: InterceptTerm | VariableTerm | InteractionTerm,
        right: VariableTerm | InteractionTerm,
    ) -> InteractionTerm:
        left_vars: tuple[str, ...]
        right_vars: tuple[str, ...]

        if isinstance(left, InterceptTerm):
            left_vars = ()
        elif isinstance(left, VariableTerm):
            left_vars = (left.name,)
        else:
            left_vars = left.variables

        right_vars = (right.name,) if isinstance(right, VariableTerm) else right.variables

        return InteractionTerm(left_vars + right_vars)

    def _combine_star(
        self,
        left: InterceptTerm | VariableTerm | InteractionTerm,
        right: VariableTerm | InteractionTerm,
    ) -> InteractionTerm:
        return self._combine_interaction(left, right)

    def _parse_random_term(self) -> RandomTerm:
        self.expect(TokenType.LPAREN)

        expr_terms: list[InterceptTerm | VariableTerm | InteractionTerm] = []
        has_intercept = True

        while self.peek().type not in (TokenType.PIPE, TokenType.DOUBLE_PIPE):
            if self.peek().type == TokenType.NUMBER:
                tok = self.advance()
                if tok.value == "1":
                    expr_terms.append(InterceptTerm())
                elif tok.value == "0":
                    has_intercept = False
            elif self.peek().type == TokenType.IDENTIFIER:
                term = self._parse_term()
                if term is not None:
                    expr_terms.append(term)
            elif self.peek().type == TokenType.PLUS:
                self.advance()
            elif self.peek().type == TokenType.MINUS:
                self.advance()
                if self.peek().type == TokenType.NUMBER and self.peek().value == "1":
                    self.advance()
                    has_intercept = False
            else:
                break

        correlated = True
        if self.peek().type == TokenType.DOUBLE_PIPE:
            self.advance()
            correlated = False
        else:
            self.expect(TokenType.PIPE)

        grouping = self._parse_grouping()
        self.expect(TokenType.RPAREN)

        return RandomTerm(
            expr=tuple(expr_terms),
            grouping=grouping,
            correlated=correlated,
            has_intercept=has_intercept,
        )

    def _parse_grouping(self) -> str | tuple[str, ...]:
        first = self.expect(TokenType.IDENTIFIER).value
        groups = [first]

        while self.peek().type == TokenType.SLASH:
            self.advance()
            groups.append(self.expect(TokenType.IDENTIFIER).value)

        if len(groups) == 1:
            return groups[0]
        return tuple(groups)


def parse_formula(formula: str) -> Formula:
    lexer = Lexer(formula)
    tokens = list(lexer.tokenize())
    parser = Parser(tokens)
    return parser.parse()
