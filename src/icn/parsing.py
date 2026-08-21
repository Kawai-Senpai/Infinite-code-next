"""Tree-sitter parsing, symbol extraction and fingerprinting.

Fingerprints are the whole basis of anchoring, so what they normalise away
matters:

    content_fingerprint   structure + identifiers, no whitespace, no comments,
                          no punctuation. Reformatting does not change it; an
                          edit to logic or to a name does.
    skeleton_fingerprint  structure only, identifiers dropped. A pure rename
                          keeps this identical, which is what lets cascade
                          step 3 tell "renamed" from "rewritten" without
                          running a full refactoring detector.
    token_signature       a bottom-k MinHash sketch of the token sequence, so
                          step 4 can measure similarity when both hashes
                          miss - at fixed size, not full text.

That pairing is the cheap version of the AST structural-signature matching
that refactoring-detection tools use (arXiv 2502.17716).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PARSERS: dict[str, Any] = {}

# Extension -> tree-sitter language name.
LANGUAGES = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".cs": "csharp",
    ".sh": "bash", ".bash": "bash",
    ".lua": "lua",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
}

# Node types that define a symbol, per language, mapped to our kind vocabulary.
SYMBOL_NODES: dict[str, dict[str, str]] = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": {
        "function_declaration": "function", "class_declaration": "class",
        "method_definition": "method", "generator_function_declaration": "function",
    },
    "typescript": {
        "function_declaration": "function", "class_declaration": "class",
        "method_definition": "method", "interface_declaration": "interface",
        "type_alias_declaration": "type", "enum_declaration": "enum",
        "abstract_class_declaration": "class",
    },
    "go": {
        "function_declaration": "function", "method_declaration": "method",
        "type_declaration": "type",
    },
    "rust": {
        "function_item": "function", "struct_item": "struct", "enum_item": "enum",
        "trait_item": "trait", "impl_item": "impl", "mod_item": "module",
    },
    "java": {
        "class_declaration": "class", "method_declaration": "method",
        "interface_declaration": "interface", "enum_declaration": "enum",
        "constructor_declaration": "constructor",
    },
    "ruby": {"method": "method", "class": "class", "module": "module", "singleton_method": "method"},
    "php": {"function_definition": "function", "class_declaration": "class", "method_declaration": "method"},
    "c": {"function_definition": "function", "struct_specifier": "struct", "enum_specifier": "enum"},
    "cpp": {
        "function_definition": "function", "class_specifier": "class",
        "struct_specifier": "struct", "namespace_definition": "namespace",
    },
    "csharp": {
        "class_declaration": "class", "method_declaration": "method",
        "interface_declaration": "interface", "struct_declaration": "struct",
    },
    "bash": {"function_definition": "function"},
    "lua": {"function_declaration": "function", "function_definition": "function"},
    "kotlin": {"class_declaration": "class", "function_declaration": "function", "object_declaration": "object"},
    "swift": {"class_declaration": "class", "function_declaration": "function", "protocol_declaration": "protocol"},
    "scala": {"class_definition": "class", "function_definition": "function", "object_definition": "object"},
}

SYMBOL_NODES["tsx"] = SYMBOL_NODES["typescript"]

# Symbols that can contain other symbols, used to build the dotted symbol path.
CONTAINER_KINDS = {"class", "interface", "struct", "module", "namespace", "trait", "impl", "object", "enum"}

COMMENT_TYPES = {"comment", "line_comment", "block_comment", "documentation_comment", "comment_block"}

# Call-expression node types, for the deterministic CALLS edges.
CALL_NODES = {
    "call", "call_expression", "method_invocation", "function_call_expression",
    "invocation_expression", "macro_invocation",
}
IMPORT_NODES = {
    "import_statement", "import_from_statement", "import_declaration", "use_declaration",
    "require_call", "preproc_include", "using_directive",
}


def language_for(path: Path) -> str | None:
    return LANGUAGES.get(path.suffix.lower())


def get_parser(lang: str):
    """Cached parser. Returns None if the grammar is unavailable, which is a
    skip-this-file condition rather than an error."""
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter_language_pack import get_parser as _get
        parser = _get(lang)
    except Exception:
        parser = None
    _PARSERS[lang] = parser
    return parser


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:32]


def _node_text(node, source: bytes) -> str:
    try:
        return source[node.start_byte:node.end_byte].decode("utf-8", "replace")
    except Exception:
        return ""


def normalize(node, source: bytes, keep_identifiers: bool) -> list[str]:
    """Flatten a subtree into a normalised token sequence.

    Only named children are walked, which drops punctuation, whitespace and
    formatting for free. Comments are dropped explicitly: a note about why the
    code exists should not be invalidated by someone rewording a docstring.
    """
    out: list[str] = []

    def walk(n) -> None:
        if n.type in COMMENT_TYPES:
            return
        named = n.named_children
        if not named:
            if keep_identifiers and n.is_named:
                out.append(f"{n.type}:{_node_text(n, source)}")
            elif n.is_named:
                out.append(n.type)
            return
        out.append("(" + n.type)
        for child in named:
            walk(child)
        out.append(")")

    walk(node)
    return out


@dataclass
class ParsedSymbol:
    name: str
    symbol_path: str
    kind: str
    lang: str
    signature: str
    ast_path: str
    start_byte: int
    end_byte: int
    line_start: int
    line_end: int
    content_fingerprint: str
    skeleton_fingerprint: str
    token_signature: str
    prev_fingerprint: str | None = None
    next_fingerprint: str | None = None
    body: str = ""
    calls: list[str] = field(default_factory=list)


def _name_of(node, source: bytes) -> str | None:
    """Best-effort symbol name. `name` field where the grammar has one, else
    the first identifier-ish descendant (C/C++ bury it under declarators)."""
    field_node = node.child_by_field_name("name")
    if field_node is not None:
        return _node_text(field_node, source).strip() or None

    declarator = node.child_by_field_name("declarator")
    probe = declarator if declarator is not None else node
    stack = list(probe.named_children)
    seen = 0
    while stack and seen < 60:
        seen += 1
        current = stack.pop(0)
        if current.type in ("identifier", "field_identifier", "type_identifier",
                            "constant", "property_identifier", "word"):
            return _node_text(current, source).strip() or None
        stack.extend(current.named_children)
    return None


def _signature(node, source: bytes, name: str) -> str:
    """First line of the declaration, trimmed. Enough to show in a capsule."""
    params = node.child_by_field_name("parameters")
    if params is not None:
        return f"{name}{_node_text(params, source)}".replace("\n", " ")[:400]
    text = _node_text(node, source)
    return text.split("\n", 1)[0].strip()[:400]


def _collect_calls(node, source: bytes, limit: int = 200) -> list[str]:
    """Callee names inside a symbol body. Deterministic but deliberately
    shallow - no type resolution, so these resolve by name later."""
    found: list[str] = []
    stack = list(node.named_children)
    while stack and len(found) < limit:
        current = stack.pop()
        if current.type in CALL_NODES:
            target = current.child_by_field_name("function") or current.child_by_field_name("name")
            if target is not None:
                text = _node_text(target, source).strip()
                if text:
                    found.append(text.split("(")[0].strip()[:120])
        stack.extend(current.named_children)
    return found


def parse_file(path: Path, source: bytes, lang: str) -> list[ParsedSymbol]:
    """Extract symbols from one file. Returns [] for anything unparseable."""
    parser = get_parser(lang)
    if parser is None:
        return []
    node_map = SYMBOL_NODES.get(lang, {})
    if not node_map:
        return []
    try:
        tree = parser.parse(source)
    except Exception:
        return []

    symbols: list[ParsedSymbol] = []
    # Each node is normalised at most once per file. Without this, every
    # symbol re-walks its previous and next sibling to build the context
    # fingerprints, so a module of N top-level functions walks the tree ~4N
    # times instead of ~N. Measured at ~40% of parse time on a large file.
    fingerprint_cache: dict[int, str] = {}

    def sibling_fingerprint(node) -> str:
        cached = fingerprint_cache.get(node.id)
        if cached is None:
            cached = _sha("".join(normalize(node, source, True)))
            fingerprint_cache[node.id] = cached
        return cached

    def visit(node, container_path: list[str], ast_path: list[str]) -> None:
        for index, child in enumerate(node.named_children):
            child_ast = ast_path + [f"{child.type}[{index}]"]
            kind = node_map.get(child.type)
            if kind is None:
                visit(child, container_path, child_ast)
                continue

            name = _name_of(child, source)
            if not name:
                visit(child, container_path, child_ast)
                continue

            symbol_path = ".".join(container_path + [name])
            content_tokens = normalize(child, source, keep_identifiers=True)
            skeleton_tokens = normalize(child, source, keep_identifiers=False)
            own_fingerprint = _sha("".join(content_tokens))
            fingerprint_cache[child.id] = own_fingerprint

            prev_fp = next_fp = None
            siblings = node.named_children
            if index > 0:
                prev_fp = sibling_fingerprint(siblings[index - 1])
            if index + 1 < len(siblings):
                next_fp = sibling_fingerprint(siblings[index + 1])

            symbols.append(ParsedSymbol(
                name=name,
                symbol_path=symbol_path,
                kind=kind,
                lang=lang,
                signature=_signature(child, source, name),
                ast_path=" > ".join(child_ast),
                start_byte=child.start_byte,
                end_byte=child.end_byte,
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                content_fingerprint=own_fingerprint,
                skeleton_fingerprint=_sha("".join(skeleton_tokens)),
                token_signature=token_signature(content_tokens),
                prev_fingerprint=prev_fp,
                next_fingerprint=next_fp,
                body=_node_text(child, source)[:4000],
                calls=_collect_calls(child, source),
            ))

            next_container = container_path + [name] if kind in CONTAINER_KINDS else container_path
            visit(child, next_container, child_ast)

    visit(tree.root_node, [], [])
    return symbols


def file_imports(source: bytes, lang: str) -> list[str]:
    """Imported module names. Used for IMPORTS edges."""
    parser = get_parser(lang)
    if parser is None:
        return []
    try:
        tree = parser.parse(source)
    except Exception:
        return []
    out: list[str] = []
    stack = [tree.root_node]
    while stack and len(out) < 300:
        node = stack.pop()
        if node.type in IMPORT_NODES:
            text = _node_text(node, source).strip()
            if text:
                out.append(text.replace("\n", " ")[:200])
            continue
        stack.extend(node.named_children)
    return out


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


SIGNATURE_SIZE = 32


def token_signature(tokens: list[str], k: int = SIGNATURE_SIZE) -> str:
    """Bottom-k MinHash sketch over token trigrams.

    Cascade step 4 needs to ask "is this plausibly the same code after an
    edit", which is a set-similarity question. Storing the full normalised
    token sequence to answer it does work, but measured on a real 4600-file
    repository it drove the store to 101MB - the sequences alone were more than
    half the database.

    A bottom-k sketch answers the same question in a fixed ~290 bytes: hash
    every trigram once, keep the k smallest, and estimate Jaccard by overlap.
    One pass, no per-permutation rehashing.
    """
    if not tokens:
        return ""
    if len(tokens) < 3:
        shingles = {" ".join(tokens)}
    else:
        shingles = {" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2)}

    hashed = sorted(
        int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=4).digest(), "big")
        for s in shingles
    )
    return ",".join(f"{h:08x}" for h in hashed[:k])


def token_similarity(a: str, b: str) -> float:
    """Estimated Jaccard similarity between two token signatures."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    left = set(a.split(","))
    right = set(b.split(","))
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0
