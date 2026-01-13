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
        ch = self.peek()
        while ch is not None and ch in " \t\n\r":
            self.advance()
            ch = self.peek()

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
                next_ch = self.peek()
                while next_ch is not None and next_ch.isdigit():
                    num += self.advance()  # type: ignore[operator]
                    next_ch = self.peek()
                yield Token(TokenType.NUMBER, num, start_pos)
            elif ch.isalpha() or ch == "_" or ch == ".":
                ident = ""
                next_ch = self.peek()
                while next_ch is not None and (next_ch.isalnum() or next_ch in "_."):
                    ident += self.advance()  # type: ignore[operator]
                    next_ch = self.peek()
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
                    if isinstance(term, tuple):
                        if term == ("no_intercept",):
                            has_intercept = False
                    else:
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


def update_formula(old_formula: Formula, new_formula_str: str) -> Formula:
    new_formula_str = new_formula_str.strip()

    if "~" not in new_formula_str:
        raise ValueError("Formula must contain '~'")

    lhs, rhs = new_formula_str.split("~", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()

    response = old_formula.response if lhs == "." else lhs

    if rhs == ".":
        return Formula(
            response=response,
            fixed=old_formula.fixed,
            random=old_formula.random,
        )

    additions: list[str] = []
    removals: list[str] = []
    random_additions: list[str] = []
    random_removals: list[str] = []
    keep_old_rhs = False

    i = 0
    rhs_clean = rhs.replace(" ", "")

    while i < len(rhs_clean):
        if rhs_clean[i] == ".":
            keep_old_rhs = True
            i += 1
        elif rhs_clean[i] == "+":
            i += 1
            if i < len(rhs_clean) and rhs_clean[i] == "(":
                paren_count = 1
                start = i
                i += 1
                while i < len(rhs_clean) and paren_count > 0:
                    if rhs_clean[i] == "(":
                        paren_count += 1
                    elif rhs_clean[i] == ")":
                        paren_count -= 1
                    i += 1
                random_additions.append(rhs_clean[start:i])
            else:
                term_start = i
                while i < len(rhs_clean) and rhs_clean[i] not in "+-":
                    i += 1
                term = rhs_clean[term_start:i]
                if term:
                    additions.append(term)
        elif rhs_clean[i] == "-":
            i += 1
            if i < len(rhs_clean) and rhs_clean[i] == "(":
                paren_count = 1
                start = i
                i += 1
                while i < len(rhs_clean) and paren_count > 0:
                    if rhs_clean[i] == "(":
                        paren_count += 1
                    elif rhs_clean[i] == ")":
                        paren_count -= 1
                    i += 1
                random_removals.append(rhs_clean[start:i])
            else:
                term_start = i
                while i < len(rhs_clean) and rhs_clean[i] not in "+-":
                    i += 1
                term = rhs_clean[term_start:i]
                if term:
                    removals.append(term)
        elif rhs_clean[i] == "(":
            paren_count = 1
            start = i
            i += 1
            while i < len(rhs_clean) and paren_count > 0:
                if rhs_clean[i] == "(":
                    paren_count += 1
                elif rhs_clean[i] == ")":
                    paren_count -= 1
                i += 1
            random_additions.append(rhs_clean[start:i])
        else:
            term_start = i
            while i < len(rhs_clean) and rhs_clean[i] not in "+-":
                i += 1
            term = rhs_clean[term_start:i]
            if term:
                additions.append(term)

    if not keep_old_rhs and not additions and not random_additions:
        return parse_formula(new_formula_str.replace(".", response))

    if keep_old_rhs:
        new_fixed_terms = list(old_formula.fixed.terms)
        has_intercept = old_formula.fixed.has_intercept
        new_random = list(old_formula.random)
    else:
        new_fixed_terms = []
        has_intercept = True
        new_random = []

    for term_str in additions:
        if term_str == "1":
            has_intercept = True
        elif term_str == "0":
            has_intercept = False
        elif ":" in term_str:
            vars_tuple = tuple(term_str.split(":"))
            interaction_term = InteractionTerm(vars_tuple)
            if interaction_term not in new_fixed_terms:
                new_fixed_terms.append(interaction_term)
        else:
            variable_term = VariableTerm(term_str)
            if variable_term not in new_fixed_terms:
                new_fixed_terms.append(variable_term)

    for term_str in removals:
        if term_str == "1":
            has_intercept = False
        elif ":" in term_str:
            vars_tuple = tuple(term_str.split(":"))
            interaction_to_remove = InteractionTerm(vars_tuple)
            new_fixed_terms = [t for t in new_fixed_terms if t != interaction_to_remove]
        else:
            variable_to_remove = VariableTerm(term_str)
            new_fixed_terms = [t for t in new_fixed_terms if t != variable_to_remove]

    for random_str in random_additions:
        temp_formula = parse_formula(f"y ~ 1 + {random_str}")
        for rt in temp_formula.random:
            if rt not in new_random:
                new_random.append(rt)

    for random_str in random_removals:
        temp_formula = parse_formula(f"y ~ 1 + {random_str}")
        for rt in temp_formula.random:
            new_random = [r for r in new_random if r != rt]

    new_fixed = FixedTerm(terms=tuple(new_fixed_terms), has_intercept=has_intercept)
    return Formula(response=response, fixed=new_fixed, random=tuple(new_random))


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

    return Formula(
        response=formula.response,
        fixed=formula.fixed,
        random=(),
    )


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

    for term in formula.fixed.terms:
        if isinstance(term, InterceptTerm):
            continue
        elif isinstance(term, VariableTerm):
            fixed_parts.append(term.name)
        elif isinstance(term, InteractionTerm):
            fixed_parts.append(":".join(term.variables))

    for rterm in formula.random:
        grouping = "/".join(rterm.grouping) if isinstance(rterm.grouping, tuple) else rterm.grouping

        if rterm.has_intercept:
            fixed_parts.append(grouping)

        for term in rterm.expr:
            if isinstance(term, InterceptTerm):
                continue
            elif isinstance(term, VariableTerm):
                fixed_parts.append(f"{grouping}:{term.name}")
            elif isinstance(term, InteractionTerm):
                interaction = ":".join(term.variables)
                fixed_parts.append(f"{grouping}:{interaction}")

    rhs = " + ".join(fixed_parts) if fixed_parts else "1"
    return f"{formula.response} ~ {rhs}"


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
        cov_map = {
            rt.grouping if isinstance(rt.grouping, str) else rt.grouping[0]: cov_type
            for rt in formula.random
        }
    else:
        for ct in cov_type.values():
            if ct not in valid_types:
                raise ValueError(f"Invalid cov_type '{ct}'. Must be one of {valid_types}")
        cov_map = cov_type

    new_random = []
    for rt in formula.random:
        group_key = rt.grouping if isinstance(rt.grouping, str) else rt.grouping[0]
        new_cov = cov_map.get(group_key, rt.cov_type)
        new_rt = RandomTerm(
            expr=rt.expr,
            grouping=rt.grouping,
            correlated=rt.correlated,
            has_intercept=rt.has_intercept,
            cov_type=new_cov,
        )
        new_random.append(new_rt)

    return Formula(
        response=formula.response,
        fixed=formula.fixed,
        random=tuple(new_random),
    )


def expandDoubleVerts(formula: str) -> str:
    """Expand || notation to explicit uncorrelated random effects.

    The || notation in lme4 specifies uncorrelated random effects.
    For example, (1 + x || group) is equivalent to (1 | group) + (0 + x | group).

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
    import re

    result = formula

    pattern = r"\(([^|()]+)\|\|([^()]+)\)"

    while True:
        match = re.search(pattern, result)
        if not match:
            break

        terms_str = match.group(1).strip()
        grouping = match.group(2).strip()

        terms = [t.strip() for t in re.split(r"[+]", terms_str)]
        terms = [t for t in terms if t]

        has_intercept = True
        if "0" in terms:
            has_intercept = False
            terms = [t for t in terms if t != "0"]
        elif "-1" in terms:
            has_intercept = False
            terms = [t for t in terms if t != "-1"]

        expanded_parts = []

        if has_intercept:
            expanded_parts.append(f"(1 | {grouping})")

        for term in terms:
            if term == "1":
                if not has_intercept:
                    expanded_parts.append(f"(1 | {grouping})")
            else:
                expanded_parts.append(f"(0 + {term} | {grouping})")

        replacement = " + ".join(expanded_parts)
        result = result[: match.start()] + replacement + result[match.end() :]

    return result


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
    import re

    formula_str = str(formula) if isinstance(formula, Formula) else formula

    pattern = r"\s*\+?\s*offset\s*\([^)]+\)\s*"
    result = re.sub(pattern, " ", formula_str)

    result = re.sub(r"\s+\+\s*$", "", result)
    result = re.sub(r"~\s*\+", "~ ", result)

    result = re.sub(r"\s+", " ", result).strip()

    return result


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
    'y ~ x * z'
    """
    if isinstance(formula, str):
        formula = parse_formula(formula)

    fixed_parts = []

    if not formula.fixed.has_intercept:
        fixed_parts.append("0")

    for term in formula.fixed.terms:
        if isinstance(term, InterceptTerm):
            continue
        elif isinstance(term, VariableTerm):
            fixed_parts.append(term.name)
        elif isinstance(term, InteractionTerm):
            fixed_parts.append(":".join(term.variables))

    rhs = " + ".join(fixed_parts) if fixed_parts else "1"
    return f"{formula.response} ~ {rhs}"


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

    random_parts = []

    for rterm in formula.random:
        grouping = "/".join(rterm.grouping) if isinstance(rterm.grouping, tuple) else rterm.grouping

        terms = []
        if rterm.has_intercept:
            terms.append("1")
        else:
            terms.append("0")

        for term in rterm.expr:
            if isinstance(term, InterceptTerm):
                continue
            elif isinstance(term, VariableTerm):
                terms.append(term.name)
            elif isinstance(term, InteractionTerm):
                terms.append(":".join(term.variables))

        expr_str = " + ".join(terms)
        separator = "||" if not rterm.correlated else "|"
        random_parts.append(f"({expr_str} {separator} {grouping})")

    return " + ".join(random_parts) if random_parts else ""


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
