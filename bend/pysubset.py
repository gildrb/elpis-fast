# Copyright (c) 2026 Gil Rodrigues
"""Restricted interpreter for the Python subset that the source links evaluate.

The source links run fragments of hash-pinned engine sources: expressions,
statement blocks, functions and plain classes. This module parses a fragment
with `ast`, accepts only an allowlist of node types and compiles the tree once
into nested Python closures, so a fragment that is evaluated in a hot loop is
parsed and checked once.

Names resolve from the caller's environment, then from a small builtin
allowlist (`BUILTINS`). `import NAME` binds only a module object the caller
passes in `modules`; `from __future__ import annotations` is accepted and has
no effect. Everything else is rejected with `SubsetError`: other imports,
`global`, `nonlocal`, `with`, `match`, generators, `async`, the walrus operator,
class keywords, every attribute name that starts with two underscores (dunder
access) and every identifier that would be name-mangled. The builtins `getattr`
and `hasattr` reject such attribute names as well, and `type` takes one
argument.

Supported nodes follow CPython semantics: evaluation order, `and` / `or`
returning an operand, chained comparisons, unpacking, scoping of functions,
classes and comprehensions, `try` / `except` / `else` / `finally`, defaults
evaluated at definition time. Every operator runs as the native Python operator
on the operands. Annotations are never evaluated and not recorded, as in a
function body; docstrings do not set `__doc__`.
"""

from __future__ import annotations

import ast
import functools
import operator
import sys
from collections.abc import Callable, Container, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Final, Literal, NoReturn, Protocol, TypeIs

__all__ = [
    "BUILTINS",
    "SubsetError",
    "compile_block",
    "compile_expr",
    "eval_expr",
    "exec_block",
]


class SubsetError(ValueError):
    """The source uses a construct outside the interpreted subset."""

    def __init__(self, what: str, lineno: int | None = None) -> None:
        """Describe the rejected construct `what`, at source line `lineno`."""
        where = "" if lineno is None else f"line {lineno}: "
        super().__init__(f"{where}{what} is outside the interpreted Python subset")


# ---------------------------------------------------------------------------
# Runtime state


@dataclass(slots=True)
class _Frame:
    """Namespaces of one executing scope.

    `glob` / `gstore` are the module namespace (read / write; `gstore` is None
    for a read-only expression environment). `local` holds the locals of a
    function or comprehension, or the namespace of a class body. `outer` is the
    nearest enclosing function or comprehension frame (free variables).
    """

    glob: Mapping[str, object]
    gstore: dict[str, object] | None
    local: dict[str, object]
    outer: _Frame | None


@dataclass(slots=True)
class _Return:
    value: object


@dataclass(frozen=True, slots=True)
class _Jump:
    keyword: str


_BREAK: Final = _Jump("break")
_CONTINUE: Final = _Jump("continue")

type _Flow = _Return | _Jump
type _Expr = Callable[[_Frame], object]
type _Stmt = Callable[[_Frame], _Flow | None]
type _Store = Callable[[_Frame, object], None]
type _Kind = Literal["module", "function", "class", "comprehension"]


class _Scope:
    """Compile-time view of one scope: its kind, local names and parent."""

    __slots__ = ("fast", "kind", "modules", "names", "parent", "qualname")

    def __init__(
        self,
        kind: _Kind,
        names: frozenset[str],
        parent: _Scope | None,
        qualname: str,
        modules: Mapping[str, object],
    ) -> None:
        self.kind: _Kind = kind
        self.names = names
        self.parent = parent
        self.qualname = qualname
        self.modules = modules
        # Function-like scopes keep their locals in their own frame
        self.fast = kind in {"function", "comprehension"}

    def child(self, kind: _Kind, names: frozenset[str], name: str) -> _Scope:
        """Return the scope of a function, class or comprehension.

        Args:
            kind: The kind of the new scope.
            names: Its local names.
            name: The name of the function or class.

        Returns:
            The new scope, nested in this one.

        """
        prefix = f"{self.qualname}.<locals>." if self.kind == "function" else ""
        if self.kind == "class":
            prefix = f"{self.qualname}."
        return _Scope(kind, names, self, prefix + name, self.modules)


# ---------------------------------------------------------------------------
# Native operators on untyped operands. The operator functions whose stubs
# accept any operands are used directly; the others narrow the operands with a
# protocol first, and the native operator then runs with full Python semantics.


class _Add(Protocol):
    def __add__(self, other: object, /) -> object: ...


class _RAdd(Protocol):
    def __radd__(self, other: object, /) -> object: ...


class _Sub(Protocol):
    def __sub__(self, other: object, /) -> object: ...


class _RSub(Protocol):
    def __rsub__(self, other: object, /) -> object: ...


class _Mul(Protocol):
    def __mul__(self, other: object, /) -> object: ...


class _RMul(Protocol):
    def __rmul__(self, other: object, /) -> object: ...


class _Mod(Protocol):
    def __mod__(self, other: object, /) -> object: ...


class _RMod(Protocol):
    def __rmod__(self, other: object, /) -> object: ...


class _Lt(Protocol):
    def __lt__(self, other: object, /) -> object: ...


class _Le(Protocol):
    def __le__(self, other: object, /) -> object: ...


class _Gt(Protocol):
    def __gt__(self, other: object, /) -> object: ...


class _Ge(Protocol):
    def __ge__(self, other: object, /) -> object: ...


class _Neg(Protocol):
    def __neg__(self) -> object: ...


class _Pos(Protocol):
    def __pos__(self) -> object: ...


class _Invert(Protocol):
    def __invert__(self) -> object: ...


class _GetItem(Protocol):
    def __getitem__(self, key: object, /) -> object: ...


class _SetItem(Protocol):
    def __setitem__(self, key: object, value: object, /) -> None: ...


class _DelItem(Protocol):
    def __delitem__(self, key: object, /) -> None: ...


# Operator support of a value's type (Python looks operators up on the type),
# by (type, method name)
_SUPPORT: Final[dict[tuple[type, str], bool]] = {}


def _supports[P](value: object, protocol: type[P], method: str) -> TypeIs[P]:
    # `protocol` is the static view of `method`; the check is the type's own
    del protocol
    key = (type(value), method)
    found = _SUPPORT.get(key)
    if found is None:
        found = _SUPPORT[key] = hasattr(key[0], method)
    return found


def _type_name(value: object) -> str:
    return type(value).__name__


def _raise_unsupported(symbol: str, a: object, b: object) -> NoReturn:
    msg = (
        f"unsupported operand type(s) for {symbol}: "
        f"'{_type_name(a)}' and '{_type_name(b)}'"
    )
    raise TypeError(msg)


def _raise_uncomparable(symbol: str, a: object, b: object) -> NoReturn:
    msg = (
        f"'{symbol}' not supported between instances of "
        f"'{_type_name(a)}' and '{_type_name(b)}'"
    )
    raise TypeError(msg)


def _raise_bad_unary(symbol: str, a: object) -> NoReturn:
    msg = f"bad operand type for unary {symbol}: '{_type_name(a)}'"
    raise TypeError(msg)


def _add(a: object, b: object) -> object:
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    if _supports(a, _Add, "__add__"):
        return a + b
    if _supports(b, _RAdd, "__radd__"):
        return a + b
    _raise_unsupported("+", a, b)


def _sub(a: object, b: object) -> object:
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    if _supports(a, _Sub, "__sub__"):
        return a - b
    if _supports(b, _RSub, "__rsub__"):
        return a - b
    _raise_unsupported("-", a, b)


def _mul(a: object, b: object) -> object:
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    if _supports(a, _Mul, "__mul__"):
        return a * b
    if _supports(b, _RMul, "__rmul__"):
        return a * b
    _raise_unsupported("*", a, b)


def _mod(a: object, b: object) -> object:
    if isinstance(a, int) and isinstance(b, int):
        return a % b
    if _supports(a, _Mod, "__mod__"):
        return a % b
    if _supports(b, _RMod, "__rmod__"):
        return a % b
    _raise_unsupported("%", a, b)


def _floordiv(a: object, b: object) -> object:
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    return operator.floordiv(a, b)


def _lt(a: object, b: object) -> object:
    if isinstance(a, int) and isinstance(b, int):
        return a < b
    if _supports(a, _Lt, "__lt__"):
        return a < b
    if _supports(b, _Gt, "__gt__"):
        return a < b
    _raise_uncomparable("<", a, b)


def _le(a: object, b: object) -> object:
    if isinstance(a, int) and isinstance(b, int):
        return a <= b
    if _supports(a, _Le, "__le__"):
        return a <= b
    if _supports(b, _Ge, "__ge__"):
        return a <= b
    _raise_uncomparable("<=", a, b)


def _gt(a: object, b: object) -> object:
    if isinstance(a, int) and isinstance(b, int):
        return a > b
    if _supports(a, _Gt, "__gt__"):
        return a > b
    if _supports(b, _Lt, "__lt__"):
        return a > b
    _raise_uncomparable(">", a, b)


def _ge(a: object, b: object) -> object:
    if isinstance(a, int) and isinstance(b, int):
        return a >= b
    if _supports(a, _Ge, "__ge__"):
        return a >= b
    if _supports(b, _Le, "__le__"):
        return a >= b
    _raise_uncomparable(">=", a, b)


def _neg(a: object) -> object:
    if isinstance(a, int):
        return -a
    if _supports(a, _Neg, "__neg__"):
        return -a
    _raise_bad_unary("-", a)


def _pos(a: object) -> object:
    if isinstance(a, int):
        return +a
    if _supports(a, _Pos, "__pos__"):
        return +a
    _raise_bad_unary("+", a)


def _invert(a: object) -> object:
    if isinstance(a, int):
        return ~a
    if _supports(a, _Invert, "__invert__"):
        return ~a
    _raise_bad_unary("~", a)


def _getitem(obj: object, key: object) -> object:
    if _supports(obj, _GetItem, "__getitem__"):
        return obj[key]
    if isinstance(obj, type):
        class_getitem = getattr(obj, "__class_getitem__", None)
        if callable(class_getitem):
            return class_getitem(key)
    msg = f"'{_type_name(obj)}' object is not subscriptable"
    raise TypeError(msg)


def _setitem(obj: object, key: object, value: object) -> None:
    if not _supports(obj, _SetItem, "__setitem__"):
        msg = f"'{_type_name(obj)}' object does not support item assignment"
        raise TypeError(msg)
    obj[key] = value


def _delitem(obj: object, key: object) -> None:
    if not _supports(obj, _DelItem, "__delitem__"):
        msg = f"'{_type_name(obj)}' object doesn't support item deletion"
        raise TypeError(msg)
    del obj[key]


# The sequence protocol: obj[0], obj[1], ... up to IndexError (or StopIteration)
def _legacy_iter(obj: _GetItem) -> Iterator[object]:
    index = 0
    while True:
        try:
            item = obj[index]
        except (IndexError, StopIteration):
            return
        yield item
        index += 1


def _iterate(value: object) -> Iterator[object]:
    # Concrete builtin types first: their isinstance check is much cheaper than
    # the Iterable ABC's
    if isinstance(value, range | list | tuple | dict | set | frozenset | str):
        return iter(value)
    if isinstance(value, Iterable):
        return iter(value)
    if _supports(value, _GetItem, "__getitem__"):
        return _legacy_iter(value)
    msg = f"'{_type_name(value)}' object is not iterable"
    raise TypeError(msg)


def _contains(container: object, item: object) -> bool:
    if isinstance(container, Container):
        return item in container
    if isinstance(container, Iterable) or _supports(container, _GetItem, "__getitem__"):
        return any(x is item or x == item for x in _iterate(container))
    msg = f"argument of type '{_type_name(container)}' is not a container or iterable"
    raise TypeError(msg)


def _in(a: object, b: object) -> object:
    return _contains(b, a)


def _not_in(a: object, b: object) -> object:
    return not _contains(b, a)


def _call(fn: object, args: list[object], kwargs: dict[str, object]) -> object:
    if not callable(fn):
        msg = f"'{_type_name(fn)}' object is not callable"
        raise TypeError(msg)
    return fn(*args, **kwargs)


type _Binary = Callable[[object, object], object]

_BINARY: Final[dict[type[ast.operator], _Binary]] = {
    ast.Add: _add,
    ast.Sub: _sub,
    ast.Mult: _mul,
    ast.Mod: _mod,
    ast.FloorDiv: _floordiv,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.MatMult: operator.matmul,
}
_INPLACE: Final[dict[type[ast.operator], _Binary]] = {
    ast.Add: operator.iadd,
    ast.Sub: operator.isub,
    ast.Mult: operator.imul,
    ast.Mod: operator.imod,
    ast.FloorDiv: operator.ifloordiv,
    ast.Div: operator.itruediv,
    ast.Pow: operator.ipow,
    ast.LShift: operator.ilshift,
    ast.RShift: operator.irshift,
    ast.BitAnd: operator.iand,
    ast.BitOr: operator.ior,
    ast.BitXor: operator.ixor,
    ast.MatMult: operator.imatmul,
}
_COMPARE: Final[dict[type[ast.cmpop], _Binary]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: _lt,
    ast.LtE: _le,
    ast.Gt: _gt,
    ast.GtE: _ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: _in,
    ast.NotIn: _not_in,
}
_UNARY: Final[dict[type[ast.unaryop], Callable[[object], object]]] = {
    ast.USub: _neg,
    ast.UAdd: _pos,
    ast.Invert: _invert,
    ast.Not: operator.not_,
}
_CONVERSIONS: Final[dict[int, Callable[[object], str]]] = {
    ord("s"): str,
    ord("r"): repr,
    ord("a"): ascii,
}


# ---------------------------------------------------------------------------
# Builtins


def _check_attribute(name: str, lineno: int | None = None) -> None:
    if name.startswith("__"):
        msg = f"attribute {name!r}"
        raise SubsetError(msg, lineno)


def _check_identifier(name: str, lineno: int | None) -> None:
    if name.startswith("__") and not name.endswith("__"):
        msg = f"name-mangled identifier {name!r}"
        raise SubsetError(msg, lineno)


def _getattr(obj: object, name: str, *default: object) -> object:
    _check_attribute(name)
    if not default:
        return getattr(obj, name)
    if len(default) == 1:
        return getattr(obj, name, default[0])
    msg = f"getattr expected at most 3 arguments, got {2 + len(default)}"
    raise TypeError(msg)


def _hasattr(obj: object, name: str) -> bool:
    _check_attribute(name)
    return hasattr(obj, name)


def _type(obj: object) -> type:
    return type(obj)


BUILTINS: Final[Mapping[str, object]] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "bytearray": bytearray,
    "classmethod": classmethod,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "float": float,
    "frozenset": frozenset,
    "getattr": _getattr,
    "hasattr": _hasattr,
    "id": id,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "staticmethod": staticmethod,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": _type,
    "zip": zip,
    "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError,
    "AttributeError": AttributeError,
    "Exception": Exception,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "LookupError": LookupError,
    "NotImplementedError": NotImplementedError,
    "OverflowError": OverflowError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
}


# ---------------------------------------------------------------------------
# Scope analysis


# The children of `node` that are evaluated in the scope `node` is in
def _scope_children(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.FunctionDef | ast.Lambda):
        args = node.args
        kw_defaults = [d for d in args.kw_defaults if d is not None]
        decorators = node.decorator_list if isinstance(node, ast.FunctionDef) else []
        return [*decorators, *args.defaults, *kw_defaults]
    if isinstance(node, ast.ClassDef):
        return [*node.decorator_list, *node.bases]
    if isinstance(node, ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp):
        return [node.generators[0].iter]
    return list(ast.iter_child_nodes(node))


# The name that `node` binds in its scope, if any
def _binding(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
        return node.id
    if isinstance(node, ast.FunctionDef | ast.ClassDef):
        return node.name
    if isinstance(node, ast.ExceptHandler):
        return node.name
    if isinstance(node, ast.alias):
        return node.asname or node.name.partition(".")[0]
    return None


# The names bound by `nodes`, not descending into nested scopes
def _bound_names(nodes: Iterable[ast.AST]) -> frozenset[str]:
    out: set[str] = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        name = _binding(node)
        if name is not None:
            out.add(name)
        stack.extend(_scope_children(node))
    return frozenset(out)


def _parameter_names(args: ast.arguments) -> list[str]:
    names = [a.arg for a in (*args.posonlyargs, *args.args)]
    if args.vararg is not None:
        names.append(args.vararg.arg)
    names.extend(a.arg for a in args.kwonlyargs)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


# ---------------------------------------------------------------------------
# Names


def _global_load(name: str) -> _Expr:
    if name not in BUILTINS:

        def load_global(f: _Frame) -> object:
            try:
                return f.glob[name]
            except KeyError:
                msg = f"name {name!r} is not defined"
                raise NameError(msg) from None

        return load_global
    builtin = BUILTINS[name]

    def load(f: _Frame) -> object:
        return f.glob.get(name, builtin)

    return load


def _local_load(name: str) -> _Expr:
    def load(f: _Frame) -> object:
        try:
            return f.local[name]
        except KeyError:
            msg = (
                f"cannot access local variable {name!r} "
                "where it is not associated with a value"
            )
            raise UnboundLocalError(msg) from None

    return load


def _up(f: _Frame, depth: int) -> _Frame:
    for _ in range(depth):
        outer = f.outer
        if outer is None:
            msg = "enclosing frame"
            raise SubsetError(msg)
        f = outer
    return f


def _free_load(name: str, depth: int) -> _Expr:
    def load(f: _Frame) -> object:
        try:
            return _up(f, depth).local[name]
        except KeyError:
            msg = (
                f"cannot access free variable {name!r} where it is not "
                "associated with a value in enclosing scope"
            )
            raise NameError(msg) from None

    return load


def _outer_load(name: str, scope: _Scope) -> _Expr:
    depth = 0
    parent = scope.parent
    while parent is not None:
        if parent.fast:
            depth += 1
            if name in parent.names:
                return _free_load(name, depth)
        parent = parent.parent
    return _global_load(name)


def _class_load(name: str, fallback: _Expr) -> _Expr:
    def load(f: _Frame) -> object:
        try:
            return f.local[name]
        except KeyError:
            return fallback(f)

    return load


def _load(name: str, scope: _Scope, lineno: int | None) -> _Expr:
    _check_identifier(name, lineno)
    if scope.kind == "module":
        return _global_load(name)
    if scope.fast:
        if name in scope.names:
            return _local_load(name)
        return _outer_load(name, scope)
    fallback = _outer_load(name, scope)
    return _class_load(name, fallback) if name in scope.names else fallback


def _store_name(name: str, scope: _Scope, lineno: int | None) -> _Store:
    _check_identifier(name, lineno)
    if scope.kind != "module":

        def store_local(f: _Frame, value: object) -> None:
            f.local[name] = value

        return store_local

    def store_global(f: _Frame, value: object) -> None:
        namespace = f.gstore
        if namespace is None:
            msg = f"assignment to {name!r} in a read-only environment"
            raise SubsetError(msg, lineno)
        namespace[name] = value

    return store_global


def _delete_name(name: str, scope: _Scope, lineno: int | None) -> _Stmt:
    _check_identifier(name, lineno)
    module = scope.kind == "module"

    def delete(f: _Frame) -> None:
        namespace = f.gstore if module else f.local
        if namespace is None or name not in namespace:
            msg = f"name {name!r} is not defined"
            raise NameError(msg)
        del namespace[name]

    return delete


# The frame that functions defined in `f` see as enclosing
def _closure_frame(f: _Frame, scope: _Scope) -> _Frame | None:
    return f if scope.fast else f.outer


# ---------------------------------------------------------------------------
# Expressions


@functools.singledispatch
def _expr(node: ast.expr, scope: _Scope) -> _Expr:
    del scope
    msg = f"expression {type(node).__name__}"
    raise SubsetError(msg, node.lineno)


def _exprs(nodes: Iterable[ast.expr], scope: _Scope) -> tuple[_Expr, ...]:
    return tuple(_expr(n, scope) for n in nodes)


@_expr.register(ast.Constant)
def _constant(node: ast.Constant, scope: _Scope) -> _Expr:
    del scope
    value = node.value

    def const(_f: _Frame) -> object:
        return value

    return const


@_expr.register(ast.Name)
def _name(node: ast.Name, scope: _Scope) -> _Expr:
    return _load(node.id, scope, node.lineno)


@_expr.register(ast.Attribute)
def _attribute(node: ast.Attribute, scope: _Scope) -> _Expr:
    _check_attribute(node.attr, node.lineno)
    obj, attr = _expr(node.value, scope), node.attr

    def load(f: _Frame) -> object:
        return getattr(obj(f), attr)

    return load


@_expr.register(ast.Subscript)
def _subscript(node: ast.Subscript, scope: _Scope) -> _Expr:
    obj, key = _expr(node.value, scope), _expr(node.slice, scope)

    def load(f: _Frame) -> object:
        return _getitem(obj(f), key(f))

    return load


def _optional(node: ast.expr | None, scope: _Scope) -> _Expr:
    if node is None:
        return _constant(ast.Constant(value=None), scope)
    return _expr(node, scope)


@_expr.register(ast.Slice)
def _slice(node: ast.Slice, scope: _Scope) -> _Expr:
    lower = _optional(node.lower, scope)
    upper = _optional(node.upper, scope)
    step = _optional(node.step, scope)

    def build(f: _Frame) -> object:
        return slice(lower(f), upper(f), step(f))

    return build


def _operator[K, V](table: Mapping[type[K], V], op: K, lineno: int) -> V:
    found = table.get(type(op))
    if found is None:
        msg = f"operator {type(op).__name__}"
        raise SubsetError(msg, lineno)
    return found


# The name if `node` reads a local of a function-like scope
def _fast_local(node: ast.expr, scope: _Scope) -> str | None:
    if isinstance(node, ast.Name) and scope.fast and node.id in scope.names:
        _check_identifier(node.id, node.lineno)
        return node.id
    return None


def _local_pair(op: _Binary, a: str, b: str, slow: _Expr) -> _Expr:
    def run(f: _Frame) -> object:
        local = f.local
        try:
            x, y = local[a], local[b]
        except KeyError:
            return slow(f)  # raises UnboundLocalError
        return op(x, y)

    return run


def _local_const(op: _Binary, a: str, value: object, slow: _Expr) -> _Expr:
    def run(f: _Frame) -> object:
        try:
            x = f.local[a]
        except KeyError:
            return slow(f)  # raises UnboundLocalError
        return op(x, value)

    return run


# `left op right`, specialized for locals and constants
def _binary(
    op: _Binary, left_node: ast.expr, right_node: ast.expr, scope: _Scope
) -> _Expr:
    left, right = _expr(left_node, scope), _expr(right_node, scope)

    def run(f: _Frame) -> object:
        return op(left(f), right(f))

    a, b = _fast_local(left_node, scope), _fast_local(right_node, scope)
    if a is not None and b is not None:
        return _local_pair(op, a, b, run)
    if isinstance(right_node, ast.Constant):
        value = right_node.value
        if a is not None:
            return _local_const(op, a, value, run)

        def run_const(f: _Frame) -> object:
            return op(left(f), value)

        return run_const
    return run


@_expr.register(ast.BinOp)
def _binop(node: ast.BinOp, scope: _Scope) -> _Expr:
    op = _operator(_BINARY, node.op, node.lineno)
    return _binary(op, node.left, node.right, scope)


@_expr.register(ast.UnaryOp)
def _unaryop(node: ast.UnaryOp, scope: _Scope) -> _Expr:
    op = _operator(_UNARY, node.op, node.lineno)
    operand = _expr(node.operand, scope)

    def run(f: _Frame) -> object:
        return op(operand(f))

    return run


@_expr.register(ast.BoolOp)
def _boolop(node: ast.BoolOp, scope: _Scope) -> _Expr:
    *heads, last = _exprs(node.values, scope)
    is_and = isinstance(node.op, ast.And)

    def run(f: _Frame) -> object:
        # The truth of the last operand is not tested: it is the result
        for value in heads:
            result = value(f)
            if bool(result) != is_and:
                return result
        return last(f)

    return run


@_expr.register(ast.Compare)
def _compare(node: ast.Compare, scope: _Scope) -> _Expr:
    ops = tuple(_operator(_COMPARE, op, node.lineno) for op in node.ops)
    first = _expr(node.left, scope)
    rest = _exprs(node.comparators, scope)
    if len(ops) == 1:
        return _binary(ops[0], node.left, node.comparators[0], scope)
    *head_ops, last_op = ops
    pairs = tuple(zip(head_ops, rest[:-1], strict=True))
    last = rest[-1]

    def run(f: _Frame) -> object:
        left = first(f)
        for op, comparator in pairs:
            right = comparator(f)
            result = op(left, right)
            if not result:
                return result
            left = right
        return last_op(left, last(f))

    return run


@_expr.register(ast.IfExp)
def _ifexp(node: ast.IfExp, scope: _Scope) -> _Expr:
    test = _expr(node.test, scope)
    body, orelse = _expr(node.body, scope), _expr(node.orelse, scope)

    def run(f: _Frame) -> object:
        return body(f) if test(f) else orelse(f)

    return run


# List / tuple / set display elements, `*x` included
def _elements(nodes: list[ast.expr], scope: _Scope) -> Callable[[_Frame], list[object]]:
    parts = tuple(
        (
            isinstance(n, ast.Starred),
            _expr(n.value if isinstance(n, ast.Starred) else n, scope),
        )
        for n in nodes
    )
    if not any(starred for starred, _ in parts):
        plain = tuple(e for _, e in parts)

        def build_plain(f: _Frame) -> list[object]:
            return [e(f) for e in plain]

        return build_plain

    def build(f: _Frame) -> list[object]:
        out: list[object] = []
        for starred, e in parts:
            if starred:
                out.extend(_iterate(e(f)))
            else:
                out.append(e(f))
        return out

    return build


@_expr.register(ast.List)
def _list(node: ast.List, scope: _Scope) -> _Expr:
    elements = _elements(node.elts, scope)

    def build(f: _Frame) -> object:
        return elements(f)

    return build


@_expr.register(ast.Tuple)
def _tuple(node: ast.Tuple, scope: _Scope) -> _Expr:
    elements = _elements(node.elts, scope)

    def build(f: _Frame) -> object:
        return tuple(elements(f))

    return build


@_expr.register(ast.Set)
def _set(node: ast.Set, scope: _Scope) -> _Expr:
    elements = _elements(node.elts, scope)

    def build(f: _Frame) -> object:
        return set(elements(f))

    return build


def _mapping_items(value: object, what: str) -> list[tuple[object, object]]:
    if not isinstance(value, Mapping):
        msg = f"{what} must be a mapping, not {_type_name(value)}"
        raise TypeError(msg)
    return list(value.items())


@_expr.register(ast.Dict)
def _dict(node: ast.Dict, scope: _Scope) -> _Expr:
    entries = tuple(
        (None if k is None else _expr(k, scope), _expr(v, scope))
        for k, v in zip(node.keys, node.values, strict=True)
    )

    def build(f: _Frame) -> object:
        out: dict[object, object] = {}
        for key, value in entries:
            if key is None:
                out.update(_mapping_items(value(f), "'**' argument"))
            else:
                k = key(f)
                out[k] = value(f)
        return out

    return build


@_expr.register(ast.JoinedStr)
def _joinedstr(node: ast.JoinedStr, scope: _Scope) -> _Expr:
    parts = _exprs(node.values, scope)

    def build(f: _Frame) -> object:
        return "".join(str(p(f)) for p in parts)

    return build


@_expr.register(ast.FormattedValue)
def _formatted(node: ast.FormattedValue, scope: _Scope) -> _Expr:
    value = _expr(node.value, scope)
    convert = _CONVERSIONS.get(node.conversion)
    spec_node = node.format_spec or ast.Constant(value="")
    spec = _expr(spec_node, scope)
    lineno = node.lineno

    def build(f: _Frame) -> object:
        v = value(f)
        s = spec(f)
        if not isinstance(s, str):
            msg = "a non-string format spec"
            raise SubsetError(msg, lineno)
        return format(v if convert is None else convert(v), s)

    return build


def _keywords(
    keywords: list[ast.keyword], scope: _Scope
) -> Callable[[_Frame], dict[str, object]]:
    parts = tuple((k.arg, _expr(k.value, scope)) for k in keywords)

    def build(f: _Frame) -> dict[str, object]:
        out: dict[str, object] = {}
        for name, value in parts:
            if name is not None:
                _add_keyword(out, name, value(f))
                continue
            for k, v in _mapping_items(value(f), "argument after **"):
                if not isinstance(k, str):
                    msg = "keywords must be strings"
                    raise TypeError(msg)
                _add_keyword(out, k, v)
        return out

    return build


def _add_keyword(out: dict[str, object], name: str, value: object) -> None:
    if name in out:
        msg = f"got multiple values for keyword argument {name!r}"
        raise TypeError(msg)
    out[name] = value


@_expr.register(ast.Call)
def _call_expr(node: ast.Call, scope: _Scope) -> _Expr:
    func = _expr(node.func, scope)
    plain = not node.keywords and not any(isinstance(a, ast.Starred) for a in node.args)
    fixed = _fixed_call(func, _exprs(node.args, scope)) if plain else None
    if fixed is not None:
        return fixed
    args = _elements(node.args, scope)
    if not node.keywords:

        def run_positional(f: _Frame) -> object:
            fn = func(f)
            return _call(fn, args(f), {})

        return run_positional
    keywords = _keywords(node.keywords, scope)

    def run(f: _Frame) -> object:
        fn = func(f)
        positional = args(f)
        return _call(fn, positional, keywords(f))

    return run


def _not_callable(fn: object) -> TypeError:
    msg = f"'{_type_name(fn)}' object is not callable"
    return TypeError(msg)


def _call0(func: _Expr) -> _Expr:
    def call(f: _Frame) -> object:
        fn = func(f)
        if not callable(fn):
            raise _not_callable(fn)
        return fn()

    return call


def _call1(func: _Expr, a: _Expr) -> _Expr:
    def call(f: _Frame) -> object:
        fn = func(f)
        x = a(f)
        if not callable(fn):
            raise _not_callable(fn)
        return fn(x)

    return call


def _call2(func: _Expr, a: _Expr, b: _Expr) -> _Expr:
    def call(f: _Frame) -> object:
        fn = func(f)
        x, y = a(f), b(f)
        if not callable(fn):
            raise _not_callable(fn)
        return fn(x, y)

    return call


def _call3(func: _Expr, a: _Expr, b: _Expr, c: _Expr) -> _Expr:
    def call(f: _Frame) -> object:
        fn = func(f)
        x, y, z = a(f), b(f), c(f)
        if not callable(fn):
            raise _not_callable(fn)
        return fn(x, y, z)

    return call


# A call with up to three plain positional arguments, without an argument list
def _fixed_call(func: _Expr, args: tuple[_Expr, ...]) -> _Expr | None:
    match args:
        case ():
            return _call0(func)
        case (a,):
            return _call1(func, a)
        case (a, b):
            return _call2(func, a, b)
        case (a, b, c):
            return _call3(func, a, b, c)
        case _:
            return None


# Comprehensions


@dataclass(slots=True)
class _Clause:
    """One `for target in source if cond ...` clause of a comprehension."""

    store: _Store
    source: _Expr
    conds: tuple[_Expr, ...]


def _produce(
    clauses: tuple[_Clause, ...], cf: _Frame, items: Iterator[object], index: int
) -> Iterator[None]:
    # Bind the targets of clause `index` and the deeper ones, once per element
    clause = clauses[index]
    last = index + 1 == len(clauses)
    for item in items:
        clause.store(cf, item)
        if not all(cond(cf) for cond in clause.conds):
            continue
        if last:
            yield None
        else:
            following = _iterate(clauses[index + 1].source(cf))
            yield from _produce(clauses, cf, following, index + 1)


type _Collect = Callable[[_Frame, Iterator[None]], object]


def _comprehension(
    generators: list[ast.comprehension],
    scope: _Scope,
    name: str,
    make: Callable[[_Scope], _Collect],
) -> _Expr:
    # `make` compiles the result in the inner scope. The first iterable is
    # evaluated in the enclosing scope, the other clauses and the result in a
    # new function-like scope (CPython semantics).
    for g in generators:
        if g.is_async:
            msg = "async comprehension"
            raise SubsetError(msg, g.target.lineno)
    first = _expr(generators[0].iter, scope)
    names = _bound_names(g.target for g in generators)
    inner = scope.child("comprehension", names, name)
    clauses = tuple(
        _Clause(_store(g.target, inner), _expr(g.iter, inner), _exprs(g.ifs, inner))
        for g in generators
    )
    collect = make(inner)

    def run(f: _Frame) -> object:
        items = _iterate(first(f))
        cf = _Frame(f.glob, f.gstore, {}, _closure_frame(f, scope))
        return collect(cf, _produce(clauses, cf, items, 0))

    return run


@_expr.register(ast.ListComp)
def _listcomp(node: ast.ListComp, scope: _Scope) -> _Expr:
    def make(inner: _Scope) -> _Collect:
        elt = _expr(node.elt, inner)

        def collect(cf: _Frame, loop: Iterator[None]) -> object:
            return [elt(cf) for _ in loop]

        return collect

    return _comprehension(node.generators, scope, "<listcomp>", make)


@_expr.register(ast.SetComp)
def _setcomp(node: ast.SetComp, scope: _Scope) -> _Expr:
    def make(inner: _Scope) -> _Collect:
        elt = _expr(node.elt, inner)

        def collect(cf: _Frame, loop: Iterator[None]) -> object:
            return {elt(cf) for _ in loop}

        return collect

    return _comprehension(node.generators, scope, "<setcomp>", make)


@_expr.register(ast.DictComp)
def _dictcomp(node: ast.DictComp, scope: _Scope) -> _Expr:
    def make(inner: _Scope) -> _Collect:
        key, value = _expr(node.key, inner), _expr(node.value, inner)

        def collect(cf: _Frame, loop: Iterator[None]) -> object:
            return {key(cf): value(cf) for _ in loop}

        return collect

    return _comprehension(node.generators, scope, "<dictcomp>", make)


@_expr.register(ast.GeneratorExp)
def _genexp(node: ast.GeneratorExp, scope: _Scope) -> _Expr:
    def make(inner: _Scope) -> _Collect:
        elt = _expr(node.elt, inner)

        def collect(cf: _Frame, loop: Iterator[None]) -> object:
            return (elt(cf) for _ in loop)

        return collect

    return _comprehension(node.generators, scope, "<genexpr>", make)


# ---------------------------------------------------------------------------
# Assignment targets


def _unpack_exact(value: object, count: int) -> tuple[object, ...]:
    if isinstance(value, tuple | list) and len(value) == count:
        return tuple(value)
    out: list[object] = []
    for item in _iterate(value):
        if len(out) == count:
            msg = f"too many values to unpack (expected {count})"
            raise ValueError(msg)
        out.append(item)
    if len(out) < count:
        msg = f"not enough values to unpack (expected {count}, got {len(out)})"
        raise ValueError(msg)
    return tuple(out)


def _unpack_starred(value: object, before: int, after: int) -> tuple[object, ...]:
    items = list(_iterate(value))
    if len(items) < before + after:
        msg = (
            f"not enough values to unpack (expected at least {before + after}, "
            f"got {len(items)})"
        )
        raise ValueError(msg)
    middle = items[before : len(items) - after]
    return (*items[:before], middle, *items[len(items) - after :])


def _store_sequence(node: ast.Tuple | ast.List, scope: _Scope) -> _Store:
    starred = [i for i, e in enumerate(node.elts) if isinstance(e, ast.Starred)]
    if len(starred) > 1:
        msg = "multiple starred expressions in assignment"
        raise SubsetError(msg, node.lineno)
    targets = tuple(
        _store(e.value if isinstance(e, ast.Starred) else e, scope) for e in node.elts
    )
    count = len(targets)
    star = starred[0] if starred else None

    def store(f: _Frame, value: object) -> None:
        if star is None:
            items = _unpack_exact(value, count)
        else:
            items = _unpack_starred(value, star, count - star - 1)
        for target, item in zip(targets, items, strict=True):
            target(f, item)

    return store


def _store(node: ast.expr, scope: _Scope) -> _Store:
    if isinstance(node, ast.Name):
        return _store_name(node.id, scope, node.lineno)
    if isinstance(node, ast.Attribute):
        _check_attribute(node.attr, node.lineno)
        obj, attr = _expr(node.value, scope), node.attr

        def store_attribute(f: _Frame, value: object) -> None:
            setattr(obj(f), attr, value)

        return store_attribute
    if isinstance(node, ast.Subscript):
        container, key = _expr(node.value, scope), _expr(node.slice, scope)

        def store_item(f: _Frame, value: object) -> None:
            _setitem(container(f), key(f), value)

        return store_item
    if isinstance(node, ast.Tuple | ast.List):
        return _store_sequence(node, scope)
    msg = f"assignment target {type(node).__name__}"
    raise SubsetError(msg, node.lineno)


# ---------------------------------------------------------------------------
# Statements


@functools.singledispatch
def _stmt(node: ast.stmt, scope: _Scope) -> _Stmt:
    del scope
    msg = f"statement {type(node).__name__}"
    raise SubsetError(msg, node.lineno)


def _block(nodes: list[ast.stmt], scope: _Scope) -> _Stmt:
    stmts = tuple(_stmt(n, scope) for n in nodes)
    if len(stmts) == 1:
        return stmts[0]

    def run(f: _Frame) -> _Flow | None:
        for s in stmts:
            flow = s(f)
            if flow is not None:
                return flow
        return None

    return run


@_stmt.register(ast.Expr)
def _expr_stmt(node: ast.Expr, scope: _Scope) -> _Stmt:
    value = _expr(node.value, scope)

    def run(f: _Frame) -> None:
        value(f)

    return run


@_stmt.register(ast.Pass)
def _pass(node: ast.Pass, scope: _Scope) -> _Stmt:
    del node, scope

    def run(_f: _Frame) -> None:
        return None

    return run


@_stmt.register(ast.Break)
def _break(node: ast.Break, scope: _Scope) -> _Stmt:
    del node, scope

    def run(_f: _Frame) -> _Flow:
        return _BREAK

    return run


@_stmt.register(ast.Continue)
def _continue(node: ast.Continue, scope: _Scope) -> _Stmt:
    del node, scope

    def run(_f: _Frame) -> _Flow:
        return _CONTINUE

    return run


@_stmt.register(ast.Return)
def _return(node: ast.Return, scope: _Scope) -> _Stmt:
    if scope.kind != "function":
        msg = "'return' outside function"
        raise SubsetError(msg, node.lineno)
    value = _optional(node.value, scope)

    def run(f: _Frame) -> _Flow:
        return _Return(value(f))

    return run


@_stmt.register(ast.Assign)
def _assign(node: ast.Assign, scope: _Scope) -> _Stmt:
    value = _expr(node.value, scope)
    first = node.targets[0]
    if len(node.targets) == 1 and isinstance(first, ast.Name) and scope.fast:
        _check_identifier(first.id, first.lineno)
        name = first.id

        def run_local(f: _Frame) -> None:
            f.local[name] = value(f)

        return run_local
    targets = tuple(_store(t, scope) for t in node.targets)
    if len(targets) == 1:
        target = targets[0]

        def run_one(f: _Frame) -> None:
            target(f, value(f))

        return run_one

    def run(f: _Frame) -> None:
        v = value(f)
        for t in targets:
            t(f, v)

    return run


@_stmt.register(ast.AnnAssign)
def _annassign(node: ast.AnnAssign, scope: _Scope) -> _Stmt:
    if node.value is None:
        return _pass(ast.Pass(), scope)
    return _assign(ast.Assign(targets=[node.target], value=node.value), scope)


def _augmented_name(node: ast.Name, scope: _Scope, op: _Binary, value: _Expr) -> _Stmt:
    load = _load(node.id, scope, node.lineno)
    store = _store_name(node.id, scope, node.lineno)

    def run(f: _Frame) -> None:
        current = load(f)
        store(f, op(current, value(f)))

    return run


def _augmented_attribute(
    node: ast.Attribute, scope: _Scope, op: _Binary, value: _Expr
) -> _Stmt:
    _check_attribute(node.attr, node.lineno)
    obj, attr = _expr(node.value, scope), node.attr

    def run(f: _Frame) -> None:
        o = obj(f)
        current = getattr(o, attr)
        setattr(o, attr, op(current, value(f)))

    return run


def _augmented_item(
    node: ast.Subscript, scope: _Scope, op: _Binary, value: _Expr
) -> _Stmt:
    container, key = _expr(node.value, scope), _expr(node.slice, scope)

    def run(f: _Frame) -> None:
        c, k = container(f), key(f)
        current = _getitem(c, k)
        _setitem(c, k, op(current, value(f)))

    return run


@_stmt.register(ast.AugAssign)
def _augassign(node: ast.AugAssign, scope: _Scope) -> _Stmt:
    op = _operator(_INPLACE, node.op, node.lineno)
    value = _expr(node.value, scope)
    target = node.target
    if isinstance(target, ast.Name):
        return _augmented_name(target, scope, op, value)
    if isinstance(target, ast.Attribute):
        return _augmented_attribute(target, scope, op, value)
    if isinstance(target, ast.Subscript):
        return _augmented_item(target, scope, op, value)
    msg = f"augmented assignment target {type(target).__name__}"
    raise SubsetError(msg, node.lineno)


def _delete_target(node: ast.expr, scope: _Scope) -> _Stmt:
    if isinstance(node, ast.Name):
        return _delete_name(node.id, scope, node.lineno)
    if isinstance(node, ast.Attribute):
        _check_attribute(node.attr, node.lineno)
        obj, attr = _expr(node.value, scope), node.attr

        def delete_attribute(f: _Frame) -> None:
            delattr(obj(f), attr)

        return delete_attribute
    if isinstance(node, ast.Subscript):
        container, key = _expr(node.value, scope), _expr(node.slice, scope)

        def delete_item(f: _Frame) -> None:
            _delitem(container(f), key(f))

        return delete_item
    if isinstance(node, ast.Tuple | ast.List):
        return _block([ast.Delete(targets=[e]) for e in node.elts], scope)
    msg = f"del target {type(node).__name__}"
    raise SubsetError(msg, node.lineno)


@_stmt.register(ast.Delete)
def _delete(node: ast.Delete, scope: _Scope) -> _Stmt:
    targets = tuple(_delete_target(t, scope) for t in node.targets)

    def run(f: _Frame) -> None:
        for t in targets:
            t(f)

    return run


def _optional_block(nodes: list[ast.stmt], scope: _Scope) -> _Stmt | None:
    return _block(nodes, scope) if nodes else None


@_stmt.register(ast.If)
def _if(node: ast.If, scope: _Scope) -> _Stmt:
    test, body = _expr(node.test, scope), _block(node.body, scope)
    orelse = _optional_block(node.orelse, scope)

    def run(f: _Frame) -> _Flow | None:
        if test(f):
            return body(f)
        return None if orelse is None else orelse(f)

    return run


def _loop_body(flow: _Flow | None) -> tuple[bool, _Flow | None]:
    # (leave the loop, flow to propagate) after one loop iteration
    if flow is None or flow is _CONTINUE:
        return False, None
    if flow is _BREAK:
        return True, None
    return True, flow


@_stmt.register(ast.For)
def _for(node: ast.For, scope: _Scope) -> _Stmt:
    source = _expr(node.iter, scope)
    target = _store(node.target, scope)
    body = _block(node.body, scope)
    orelse = _optional_block(node.orelse, scope)

    def run(f: _Frame) -> _Flow | None:
        for item in _iterate(source(f)):
            target(f, item)
            flow = body(f)
            if flow is None:
                continue
            leave, flow = _loop_body(flow)
            if leave:
                return flow
        return None if orelse is None else orelse(f)

    return run


@_stmt.register(ast.While)
def _while(node: ast.While, scope: _Scope) -> _Stmt:
    test, body = _expr(node.test, scope), _block(node.body, scope)
    orelse = _optional_block(node.orelse, scope)

    def run(f: _Frame) -> _Flow | None:
        while test(f):
            leave, flow = _loop_body(body(f))
            if leave:
                return flow
        return None if orelse is None else orelse(f)

    return run


def _exception(value: object, what: str) -> BaseException:
    if isinstance(value, type) and issubclass(value, BaseException):
        return value()
    if isinstance(value, BaseException):
        return value
    msg = f"{what} must derive from BaseException"
    raise TypeError(msg)


@_stmt.register(ast.Raise)
def _raise(node: ast.Raise, scope: _Scope) -> _Stmt:
    if node.exc is None:

        def reraise(_f: _Frame) -> None:
            active = sys.exception()
            if active is None:
                msg = "No active exception to reraise"
                raise RuntimeError(msg)
            raise active

        return reraise
    exc = _expr(node.exc, scope)
    if node.cause is None:

        def run(f: _Frame) -> None:
            raise _exception(exc(f), "exceptions")

        return run
    cause = _expr(node.cause, scope)

    def run_from(f: _Frame) -> None:
        error = _exception(exc(f), "exceptions")
        c = cause(f)
        raise error from (None if c is None else _exception(c, "exception causes"))

    return run_from


@_stmt.register(ast.Assert)
def _assert(node: ast.Assert, scope: _Scope) -> _Stmt:
    test = _expr(node.test, scope)
    message = None if node.msg is None else _expr(node.msg, scope)

    def run(f: _Frame) -> None:
        if not test(f):
            raise AssertionError if message is None else AssertionError(message(f))

    return run


# The classes an `except` clause matches, as CPython checks them
def _exception_classes(value: object) -> tuple[type[BaseException], ...]:
    if isinstance(value, type) and issubclass(value, BaseException):
        return (value,)
    if isinstance(value, tuple):
        out: list[type[BaseException]] = []
        for item in value:
            out.extend(_exception_classes(item))
        return tuple(out)
    msg = "catching classes that do not inherit from BaseException is not allowed"
    raise TypeError(msg)


class _Handler:
    """One compiled `except` clause."""

    __slots__ = ("bind", "body", "matches", "unbind")

    def __init__(self, node: ast.ExceptHandler, scope: _Scope) -> None:
        kind = None if node.type is None else _expr(node.type, scope)
        self.matches: Callable[[_Frame, BaseException], bool] = _matcher(kind)
        self.body = _block(node.body, scope)
        name = node.name
        self.bind = None if name is None else _store_name(name, scope, node.lineno)
        self.unbind = None if name is None else _unbind_name(name, scope)


# Remove `name` if it is bound (the end of an `except ... as name` clause)
def _unbind_name(name: str, scope: _Scope) -> _Stmt:
    module = scope.kind == "module"

    def unbind(f: _Frame) -> None:
        namespace = f.gstore if module else f.local
        if namespace is not None:
            namespace.pop(name, None)

    return unbind


def _matcher(kind: _Expr | None) -> Callable[[_Frame, BaseException], bool]:
    def matches(f: _Frame, error: BaseException) -> bool:
        return kind is None or isinstance(error, _exception_classes(kind(f)))

    return matches


def _handle(
    handlers: tuple[_Handler, ...], f: _Frame, error: BaseException
) -> tuple[bool, _Flow | None]:
    # Run the first matching handler; return (handled, flow)
    for handler in handlers:
        if not handler.matches(f, error):
            continue
        if handler.bind is not None:
            handler.bind(f, error)
        try:
            return True, handler.body(f)
        finally:
            if handler.unbind is not None:
                handler.unbind(f)
    return False, None


@_stmt.register(ast.Try)
def _try(node: ast.Try, scope: _Scope) -> _Stmt:
    body = _block(node.body, scope)
    handlers = tuple(_Handler(h, scope) for h in node.handlers)
    orelse = _optional_block(node.orelse, scope)
    final = _optional_block(node.finalbody, scope)

    def guarded(f: _Frame) -> _Flow | None:
        try:
            flow = body(f)
        except BaseException as error:
            handled, flow = _handle(handlers, f, error)
            if not handled:
                raise
            return flow
        if flow is not None or orelse is None:
            return flow
        return orelse(f)

    if final is None:
        return guarded

    def run(f: _Frame) -> _Flow | None:
        try:
            flow = guarded(f)
        except BaseException:
            override = final(f)
            if override is not None:
                return override
            raise
        override = final(f)
        return flow if override is None else override

    return run


@_stmt.register(ast.Import)
def _import(node: ast.Import, scope: _Scope) -> _Stmt:
    stores: list[tuple[_Store, object]] = []
    for alias in node.names:
        if "." in alias.name or alias.name not in scope.modules:
            msg = f"import of {alias.name!r} (not a module passed by the caller)"
            raise SubsetError(msg, node.lineno)
        name = alias.asname or alias.name
        module = scope.modules[alias.name]
        stores.append((_store_name(name, scope, node.lineno), module))

    def run(f: _Frame) -> None:
        for store, module in stores:
            store(f, module)

    return run


@_stmt.register(ast.ImportFrom)
def _import_from(node: ast.ImportFrom, scope: _Scope) -> _Stmt:
    names = [a.name for a in node.names]
    if node.module != "__future__" or names != ["annotations"]:
        msg = f"from {node.module} import {', '.join(names)}"
        raise SubsetError(msg, node.lineno)
    return _pass(ast.Pass(), scope)


# Functions and classes


class _Binder:
    """Binds call arguments to the parameters of one function, as CPython does."""

    __slots__ = (
        "defaults",
        "kwarg",
        "kwdefaults",
        "kwonly",
        "name",
        "positional",
        "posonly",
        "simple",
        "vararg",
    )

    def __init__(
        self,
        args: ast.arguments,
        name: str,
        defaults: tuple[object, ...],
        kwdefaults: dict[str, object],
    ) -> None:
        self.name = name
        self.positional = tuple(a.arg for a in (*args.posonlyargs, *args.args))
        self.posonly = len(args.posonlyargs)
        self.vararg = None if args.vararg is None else args.vararg.arg
        self.kwonly = tuple(a.arg for a in args.kwonlyargs)
        self.kwarg = None if args.kwarg is None else args.kwarg.arg
        self.defaults = defaults
        self.kwdefaults = kwdefaults
        self.simple = not (defaults or self.vararg or self.kwonly or self.kwarg)

    def bind(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> dict[str, object]:
        # The locals of a call with `args` and `kwargs`
        if self.simple and not kwargs and len(args) == len(self.positional):
            return dict(zip(self.positional, args, strict=True))
        local = self._bind_positional(args)
        extra = self._bind_keywords(local, kwargs)
        self._fill_defaults(local)
        if self.kwarg is not None:
            local[self.kwarg] = extra
        return local

    def _bind_positional(self, args: tuple[object, ...]) -> dict[str, object]:
        count = len(self.positional)
        if len(args) > count and self.vararg is None:
            msg = (
                f"{self.name}() takes {count} positional argument"
                f"{'' if count == 1 else 's'} but {len(args)} were given"
            )
            raise TypeError(msg)
        local = dict(zip(self.positional, args[:count], strict=False))
        if self.vararg is not None:
            local[self.vararg] = tuple(args[count:])
        return local

    def _bind_keywords(
        self, local: dict[str, object], kwargs: dict[str, object]
    ) -> dict[str, object]:
        extra: dict[str, object] = {}
        named = set(self.positional[self.posonly :]) | set(self.kwonly)
        for key, value in kwargs.items():
            if key not in named:
                if self.kwarg is None:
                    msg = f"{self.name}() got an unexpected keyword argument {key!r}"
                    raise TypeError(msg)
                extra[key] = value
            elif key in local:
                msg = f"{self.name}() got multiple values for argument {key!r}"
                raise TypeError(msg)
            else:
                local[key] = value
        return extra

    def _fill_defaults(self, local: dict[str, object]) -> None:
        first_default = len(self.positional) - len(self.defaults)
        for index, param in enumerate(self.positional):
            if param in local:
                continue
            if index < first_default:
                msg = f"{self.name}() missing required positional argument: {param!r}"
                raise TypeError(msg)
            local[param] = self.defaults[index - first_default]
        for param in self.kwonly:
            if param in local:
                continue
            if param not in self.kwdefaults:
                msg = f"{self.name}() missing required keyword-only argument: {param!r}"
                raise TypeError(msg)
            local[param] = self.kwdefaults[param]


def _check_parameters(args: ast.arguments, lineno: int) -> None:
    for name in _parameter_names(args):
        _check_identifier(name, lineno)


def _defaults(
    args: ast.arguments, scope: _Scope
) -> Callable[[_Frame], tuple[tuple[object, ...], dict[str, object]]]:
    positional = _exprs(args.defaults, scope)
    keyword = tuple(
        (a.arg, _expr(d, scope))
        for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True)
        if d is not None
    )

    def evaluate(f: _Frame) -> tuple[tuple[object, ...], dict[str, object]]:
        return tuple(d(f) for d in positional), {k: d(f) for k, d in keyword}

    return evaluate


def _make_function(
    binder: _Binder, body: _Stmt, f: _Frame, closure: _Frame | None, qualname: str
) -> Callable[..., object]:
    glob, gstore = f.glob, f.gstore

    def function(*args: object, **kwargs: object) -> object:
        flow = body(_Frame(glob, gstore, binder.bind(args, kwargs), closure))
        if isinstance(flow, _Return):
            return flow.value
        if flow is not None:
            msg = f"'{flow.keyword}' outside loop"
            raise SubsetError(msg)
        return None

    function.__name__ = binder.name
    function.__qualname__ = qualname
    return function


def _decorate(decorators: list[object], value: object) -> object:
    for decorator in reversed(decorators):
        value = _call(decorator, [value], {})
    return value


@_stmt.register(ast.FunctionDef)
def _function_def(node: ast.FunctionDef, scope: _Scope) -> _Stmt:
    if node.type_params:
        msg = "type parameters"
        raise SubsetError(msg, node.lineno)
    _check_parameters(node.args, node.lineno)
    decorators = _exprs(node.decorator_list, scope)
    defaults = _defaults(node.args, scope)
    names = frozenset(_parameter_names(node.args)) | _bound_names(node.body)
    inner = scope.child("function", names, node.name)
    body = _block(node.body, inner)
    store = _store_name(node.name, scope, node.lineno)
    args, name, qualname = node.args, node.name, inner.qualname

    def run(f: _Frame) -> None:
        decos = [d(f) for d in decorators]
        positional, keyword = defaults(f)
        binder = _Binder(args, name, positional, keyword)
        function = _make_function(binder, body, f, _closure_frame(f, scope), qualname)
        store(f, _decorate(decos, function))

    return run


@_expr.register(ast.Lambda)
def _lambda(node: ast.Lambda, scope: _Scope) -> _Expr:
    _check_parameters(node.args, node.lineno)
    defaults = _defaults(node.args, scope)
    names = frozenset(_parameter_names(node.args))
    inner = scope.child("function", names, "<lambda>")
    result = _expr(node.body, inner)
    args, qualname = node.args, inner.qualname

    def body(f: _Frame) -> _Flow:
        return _Return(result(f))

    def run(f: _Frame) -> object:
        positional, keyword = defaults(f)
        binder = _Binder(args, "<lambda>", positional, keyword)
        return _make_function(binder, body, f, _closure_frame(f, scope), qualname)

    return run


def _class_bases(bases: tuple[_Expr, ...], f: _Frame) -> tuple[type, ...]:
    out: list[type] = []
    for base in bases:
        value = base(f)
        if not isinstance(value, type):
            msg = f"class base of type {_type_name(value)}"
            raise SubsetError(msg)
        out.append(value)
    return tuple(out)


@_stmt.register(ast.ClassDef)
def _class_def(node: ast.ClassDef, scope: _Scope) -> _Stmt:
    if node.keywords or node.type_params:
        msg = "class keywords or type parameters"
        raise SubsetError(msg, node.lineno)
    _check_identifier(node.name, node.lineno)
    decorators = _exprs(node.decorator_list, scope)
    bases = _exprs(node.bases, scope)
    inner = scope.child("class", _bound_names(node.body), node.name)
    body = _block(node.body, inner) if node.body else None
    store = _store_name(node.name, scope, node.lineno)
    name, qualname = node.name, inner.qualname

    def run(f: _Frame) -> None:
        decos = [d(f) for d in decorators]
        resolved = _class_bases(bases, f)
        namespace: dict[str, object] = {
            "__module__": f.glob.get("__name__", "builtins"),
            "__qualname__": qualname,
        }
        if body is not None:
            body(_Frame(f.glob, f.gstore, namespace, _closure_frame(f, scope)))
        store(f, _decorate(decos, type(name, resolved, namespace)))

    return run


# ---------------------------------------------------------------------------
# Public API


def _parse(source: str, mode: Literal["eval", "exec"]) -> ast.AST:
    try:
        return ast.parse(source, mode=mode)
    except SyntaxError as error:
        msg = f"unparsable source ({error.msg})"
        raise SubsetError(msg, error.lineno) from error


@functools.cache
def compile_expr(source: str) -> Callable[[Mapping[str, object]], object]:
    """Compile one expression for repeated evaluation.

    Args:
        source: The expression text.

    Returns:
        A function that evaluates the expression with names from its mapping
        argument (then `BUILTINS`).

    Raises:
        SubsetError: The source does not parse or leaves the subset.

    """
    # As eval() does with a string, strip leading and trailing spaces and tabs
    tree = _parse(source.strip(" \t"), "eval")
    if not isinstance(tree, ast.Expression):
        msg = "expression mode"
        raise SubsetError(msg)
    scope = _Scope("module", frozenset(), None, "", {})
    code = _expr(tree.body, scope)

    def evaluate(env: Mapping[str, object]) -> object:
        return code(_Frame(env, None, {}, None))

    return evaluate


def eval_expr(source: str, env: Mapping[str, object]) -> object:
    """Evaluate one expression.

    Args:
        source: The expression text.
        env: The names the expression reads (before `BUILTINS`).

    Returns:
        The value of the expression.

    """
    return compile_expr(source)(env)


def compile_block(
    source: str | ast.Module, modules: Mapping[str, object] | None = None
) -> Callable[[dict[str, object]], None]:
    """Compile a statement block (a module body) for repeated execution.

    Args:
        source: The statements, as text or as a parsed module.
        modules: The objects that `import NAME` binds, by module name.

    Returns:
        A function that runs the statements with its argument as the module
        namespace: names are read from it (then `BUILTINS`) and written to it.

    Raises:
        SubsetError: The source does not parse or leaves the subset.

    """
    tree = source if isinstance(source, ast.Module) else _parse(source, "exec")
    if not isinstance(tree, ast.Module):
        msg = "exec mode"
        raise SubsetError(msg)
    scope = _Scope("module", frozenset(), None, "", modules or {})
    body = _block(tree.body, scope) if tree.body else None

    def run(env: dict[str, object]) -> None:
        if body is None:
            return
        flow = body(_Frame(env, env, {}, None))
        if isinstance(flow, _Jump):
            msg = f"'{flow.keyword}' outside loop"
            raise SubsetError(msg)

    return run


def exec_block(
    source: str | ast.Module,
    env: dict[str, object],
    modules: Mapping[str, object] | None = None,
) -> None:
    """Run a statement block (a module body) in `env`.

    Args:
        source: The statements, as text or as a parsed module.
        env: The module namespace: names are read from it (then `BUILTINS`)
            and the statements' definitions are written to it.
        modules: The objects that `import NAME` binds, by module name.

    """
    compile_block(source, modules)(env)
