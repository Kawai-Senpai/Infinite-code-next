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

# What this module extracts, as a number stored on every file row.
#
# Bump it whenever parse_file or file_import_specs starts producing something
# it did not before. The indexer compares the stored value against this one and
# re-stores any file that is behind, which is the only reliable way to migrate:
# file content is unchanged, so a content hash cannot detect that the EXTRACTOR
# moved. Using some other column as a proxy for staleness - "imports_raw is
# NULL" - survives exactly one version step, then silently strands every row
# that a partial run already touched.
#
#   1  symbols, fingerprints, calls
#   2  structured import specifiers (module/level/probe/deferred)
#   3  decorators, and so entry points
#   4  per-language extraction: JS/TS import sources rather than imported
#      names, require()/import(), Rust mod and sibling attributes, Java and
#      Kotlin annotations grouped under `modifiers`, Go method receivers,
#      JS/TS arrow-function and function-expression declarations
#   5  correctness: operators and keywords included in fingerprints, calls
#      scope-aware and in source order, comment-stable ast_path and context
#      fingerprints, JS re-exports, dynamic import() and TYPE_CHECKING as
#      deferred, grouped Go type specs
#   6  symbols and calls for any language with a tags.scm, so a language
#      without hand-written rules yields real structure rather than nothing
#   7  module-level calls owned by a `<module>` pseudo-symbol, and receiver
#      type bindings recorded per symbol so a call through a local variable
#      or a field can be resolved to a method
#   8  block-level clone fragments (extract_fragments), with alpha, content and
#      skeleton fingerprints per fragment. Symbol fingerprints are unchanged by
#      this step, so no anchor moves; the bump exists to backfill the new
#      clone_fragments rows for files whose content did not change.
EXTRACT_VERSION = 8

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
    ".kt": "kotlin", ".kts": "kotlin",
    ".swift": "swift",
    ".scala": "scala", ".sc": "scala",
    # Pinned because the hand-written rules above name them. Everything else
    # falls through to the language pack's own detection in language_for.
    ".mts": "typescript", ".cts": "typescript",
    ".pyw": "python",
    ".hxx": "cpp", ".ipp": "cpp", ".tpp": "cpp",
}

# Node types that define a symbol, per language, mapped to our kind vocabulary.
SYMBOL_NODES: dict[str, dict[str, str]] = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": {
        "function_declaration": "function", "class_declaration": "class",
        "method_definition": "method", "generator_function_declaration": "function",
        "variable_declarator": "function",
    },
    "typescript": {
        "function_declaration": "function", "class_declaration": "class",
        "method_definition": "method", "interface_declaration": "interface",
        "type_alias_declaration": "type", "enum_declaration": "enum",
        "abstract_class_declaration": "class",
        "variable_declarator": "function",
    },
    "go": {
        "function_declaration": "function", "method_declaration": "method",
        # `type_spec`, not the `type_declaration` wrapper around it. Go allows
        # `type ( A struct{}; B interface{} )`, where one declaration holds
        # many specs and each owns its own name - matching the wrapper found
        # only the first and silently dropped every other type in the group.
        "type_spec": "type", "type_alias": "type",
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

# What a `variable_declarator` has to be bound to before it counts as a symbol.
# `const total = 1` is a variable and stays out of the symbol table; `const
# Panel = () => <div/>` is a function declaration in all but spelling, and in
# modern JavaScript and React it is the dominant one - a component file written
# this way produced no symbols at all, so nothing called it and nothing it
# called was reachable from it.
VALUE_FUNCTION_NODES = {"arrow_function", "function_expression", "function"}

# Nodes that bind a name to a value, where the value may be a function. Their
# body lives one level down, under the `value` field.
BINDING_NODES = {"variable_declarator", "field_definition",
                 "public_field_definition", "assignment_expression"}

# Symbols that can contain other symbols, used to build the dotted symbol path.
CONTAINER_KINDS = {"class", "interface", "struct", "module", "namespace", "trait", "impl", "object", "enum"}

COMMENT_TYPES = {"comment", "line_comment", "block_comment", "documentation_comment", "comment_block"}

# Call-expression node types, for the deterministic CALLS edges.
CALL_NODES = {
    "call", "call_expression", "method_invocation", "function_call_expression",
    "invocation_expression", "macro_invocation",
}
# Nodes whose body only runs when called. An import inside one of these cannot
# affect module-initialisation order, which is what makes an import cycle a
# real problem rather than a formality - a function-level import is the
# standard way to break a cycle, so counting it as one reports the fix as the
# fault.
FUNCTION_BODY_NODES = {
    "function_definition", "function_declaration", "method_definition",
    "method_declaration", "function_item", "arrow_function", "lambda",
    "generator_function_declaration", "constructor_declaration",
    "function_expression", "local_function_statement",
}

IMPORT_NODES = {
    "import_statement", "import_from_statement", "import_declaration", "use_declaration",
    "require_call", "preproc_include", "using_directive",
    # Kotlin wraps each import in an `import_header`; PHP uses its own node for
    # `use App\Store;`. Both were missing, so neither language produced a single
    # import edge - the whole module graph for a Kotlin or PHP repository was
    # absent rather than merely incomplete.
    "import_header", "namespace_use_declaration",
    # `mod auth;` is a file reference. `mod tests { .. }` is a definition and is
    # rejected in _rust_module_specs, which is why this cannot be decided here.
    "mod_item",
}

# Import nodes whose subtree still has to be walked. Everything else in
# IMPORT_NODES is a leaf statement, but a Rust `mod name { .. }` block holds
# real code, and an inline module full of `use` statements would be skipped
# whole if the walk stopped at the module.
IMPORT_NODES_WITH_BODIES = {"mod_item"}

# Functions whose first string argument names a module. These are ordinary call
# expressions, not import statements, so no grammar marks them - but CommonJS
# is how a large share of real JavaScript states its dependencies, and without
# them a `require`-based file has no imports at all.
REQUIRE_CALLS = {"require", "import", "require_relative", "requirejs"}

# Languages whose import statements name the module in a `source` field.
JS_IMPORT_LANGS = {"javascript", "typescript", "tsx"}

# `export { X } from "./m"` is an import with an export's spelling. Not in
# IMPORT_NODES because the node is not a leaf: `export function f() { ... }`
# is the same type and its body still has to be walked.
EXPORT_IMPORT_NODES = {"export_statement"}


def language_for(path: Path) -> str | None:
    """Language for a file, by extension.

    The table above is an override, not the whole detector: it pins the
    languages this module has hand-written extraction rules for, so a grammar
    rename upstream cannot silently repoint `.ts`. Anything it does not name
    falls through to the language pack, which knows several hundred more
    extensions than are worth restating here - `.mts`, `.cts`, `.kts`, `.pyw`,
    `.hxx` and the rest were all simply unrecognised.

    Recognising a file is not the same as understanding it. A language with no
    entry in SYMBOL_NODES parses and yields no symbols; see `support_tier`.
    """
    explicit = LANGUAGES.get(path.suffix.lower())
    if explicit is not None:
        return explicit
    try:
        from tree_sitter_language_pack import detect_language_from_path
        detected = detect_language_from_path(str(path))
    except Exception:
        return None
    return detected or None


def get_parser(lang: str):
    """Cached parser. Returns None if the grammar is unavailable, which is a
    skip-this-file condition rather than an error.

    Only successes are cached. The language pack loads grammars on demand, so a
    failure can be transient - a partial download, a busy cache - and caching
    it disabled that language for the life of the process. The cost of retrying
    a genuinely missing grammar is one failed import per file; the cost of
    caching a transient failure was an entire language silently missing from
    the index.
    """
    parser = _PARSERS.get(lang)
    if parser is not None:
        return parser
    try:
        from tree_sitter_language_pack import get_parser as _get
        parser = _get(lang)
    except Exception:
        return None
    _PARSERS[lang] = parser
    return parser


def support_tier(lang: str) -> str:
    """How much this module actually understands about a language.

    Parsing several hundred grammars is not the same as extracting from them,
    and reporting the larger number would mean "supporting" a language by
    returning an empty symbol list - worse than saying support is partial.

        full     hand-written symbol, import and call rules. These carry the
                 things a generic query cannot know: Go receivers, Rust impl
                 blocks, out-of-line C++ definitions, decorators.
        tags     symbols and calls derived from tree-sitter's own tags.scm.
                 Real extraction, but no import resolution and no per-language
                 corrections.
        parse    a grammar exists and nothing is extracted from it.
        none     no grammar.
    """
    if lang in SYMBOL_NODES:
        return "full"
    if get_parser(lang) is None:
        return "none"
    return "tags" if _tags_query(lang) is not None else "parse"


# tags.scm capture name -> our kind vocabulary. Upstream defines these per
# grammar, so this is a reading of what the language's own maintainers call
# each construct rather than a guess from node names.
TAG_KINDS = {
    "definition.class": "class", "definition.function": "function",
    "definition.method": "method", "definition.module": "module",
    "definition.interface": "interface", "definition.type": "type",
    "definition.struct": "struct", "definition.enum": "enum",
    "definition.trait": "trait", "definition.union": "struct",
    "definition.constant": "constant", "definition.macro": "function",
    "definition.operator": "function", "definition.constructor": "constructor",
}

_TAGS_CACHE: dict[str, Any] = {}


def _split_patterns(text: str) -> list[str]:
    """Top-level s-expressions in a tags.scm, one per pattern."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    in_string = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            if depth == 0 and buf:
                out.append("\n".join(buf))
                buf = []
            continue
        buf.append(line)
        for char in line:
            if char == '"':
                in_string = not in_string
            elif not in_string:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
        if depth == 0 and buf:
            out.append("\n".join(buf))
            buf = []
    if buf:
        out.append("\n".join(buf))
    return out


def _tags_query(lang: str):
    """Compiled definition patterns from the language's own tags.scm, or None.

    Patterns are compiled one at a time and the failures discarded. The
    bundled query and the compiled grammar are versioned separately, so a
    query can name a node the grammar no longer has - dart's does - and
    compiling the file as a unit then yields nothing at all for a language
    whose other four patterns are perfectly good.
    """
    if lang in _TAGS_CACHE:
        return _TAGS_CACHE[lang]
    _TAGS_CACHE[lang] = None
    try:
        import tree_sitter
        from tree_sitter_language_pack import get_language, get_tags_query
        language = get_language(lang)
        source = get_tags_query(lang)
    except Exception:
        return None
    if not source:
        return None

    usable = []
    for pattern in _split_patterns(source):
        if "@definition." not in pattern and "@reference.call" not in pattern:
            continue
        try:
            tree_sitter.Query(language, pattern)
        except Exception:
            continue
        usable.append(pattern)
    if not usable:
        return None
    try:
        _TAGS_CACHE[lang] = tree_sitter.Query(language, "\n".join(usable))
    except Exception:
        _TAGS_CACHE[lang] = None
    return _TAGS_CACHE[lang]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:32]


def _node_text(node, source: bytes) -> str:
    try:
        return source[node.start_byte:node.end_byte].decode("utf-8", "replace")
    except Exception:
        return ""


# Punctuation that carries no meaning of its own. Tree-sitter calls every
# literal token in the grammar "anonymous", which lumps `+`, `===` and `await`
# in with `(`, `,` and `;`. Dropping all of them - which this did - made
# `a + b` and `a - b` produce the same fingerprint, so changing an operator
# did not invalidate an anchor. Dropping none of them makes a fingerprint
# sensitive to a trailing comma or an optional semicolon, which is the thing
# fingerprints exist to ignore. So only the grouping and separating tokens go.
# Every token here is redundant with the tree SHAPE: `(a + b) * c` and
# `a + b * c` already nest differently, and a separator is implied by the child
# count. `<` and `>` deliberately are NOT here - the tree gives `a < b` and
# `a > b` the same shape, so dropping them let a comparison flip go unnoticed.
IGNORED_ANONYMOUS = {
    "(", ")", "[", "]", "{", "}",
    ",", ";", ":", ".", "->", "=>", "::",
}


# Fields whose identifier child sits on the right of a dot. NiCad's consistent
# renaming renumbers every identifier including the receiver, so `cache.get(k)`
# and `db.get(u)` both become `V1.get(V2)`. That is correct Type-2 behaviour and
# we keep it.
#
# Being on the right of a dot is NOT on its own a reason to preserve a name.
# Only a CALLED attribute is preserved (see is_called): `.strip()` and `.match()`
# are API surface, so collapsing them would make a read and a delete hash alike,
# but `req.email` and `data.address` are data fields whose names vary freely
# between callers. Preserving those was measured to break the commonest real
# case this feature exists to catch - one validation block copied between two
# modules that spell the same field differently.
#
# `attribute`  Python  a.b          -> b
# `field`      Go, Java, JS, Rust   -> the selected field
# `property`   JS/TS member exprs
ATTRIBUTE_FIELDS = {"attribute", "field", "property"}

# Leaf node types that NAME an attribute rather than bind a variable. These are
# the second route to the same distinction: Python marks `.get` by its parent's
# field name, while JS and Go give it a distinct leaf type outright. Both are
# checked, because relying on either alone silently renames method names in
# half the supported languages.
ATTRIBUTE_TYPES = {
    "property_identifier",   # JS/TS   o.attr
    "field_identifier",      # Go, Rust  o.Attr
}

# Leaves that bind or reference a variable, and so get a placeholder.
# `type_identifier` is deliberately absent: a type is API surface like an
# attribute, and renaming it would make `List[int]` and `Dict[str]` collide.
IDENTIFIER_TYPES = {
    "identifier",
    "word",                  # some grammars' generic identifier
    "simple_identifier",     # Kotlin
    "shorthand_property_identifier",
}

# Literal leaves, folded to their bare type so a changed constant does not
# split a clone class: `timeout = 30` and `timeout = 60` are the same logic.
# Matched by suffix as well as by name, because every grammar spells these
# differently (integer / int_literal / decimal_integer_literal / ...).
LITERAL_TYPES = {
    "integer", "float", "number", "true", "false", "none", "null", "nil",
    "string_content", "string_fragment", "interpreted_string_literal_content",
    "raw_string_literal", "character", "boolean_literal",
}

LITERAL_SUFFIXES = ("_literal", "_literal_content")


def _is_literal(node_type: str) -> bool:
    return node_type in LITERAL_TYPES or node_type.endswith(LITERAL_SUFFIXES)


def alpha_normalize(node, source: bytes) -> list[str]:
    """Consistent renaming: each distinct identifier becomes V1, V2, ... in
    first-appearance order, while identical names keep identical placeholders.

    This is the middle tier of the three canonical forms, and the only one
    trustworthy enough to drive a refactoring recommendation:

        content   identifiers kept       `total` != `result`
        alpha     identifiers renumbered `total` == `result`, `V1=V2` != `V1=V1`
        skeleton  identifiers erased     `V1=V2` == `V1=V1`

    Skeleton is deliberately permissive because the anchor cascade adjudicates
    afterwards with token_similarity. Nothing adjudicates a clone report, so
    alpha has to carry its own precision. Consistent renaming is what buys it:
    a fragment that assigns a variable to itself and one that assigns from
    another variable are structurally different programs, and skeleton cannot
    tell them apart.

    Attribute names survive (see ATTRIBUTE_FIELDS); literals are folded to
    their node type so `timeout = 30` and `timeout = 60` are one clone, which
    is what makes a "same logic, different constant" duplicate findable.
    """
    out: list[str] = []
    names: dict[str, str] = {}

    def placeholder(text: str) -> str:
        slot = names.get(text)
        if slot is None:
            slot = f"V{len(names) + 1}"
            names[text] = slot
        return slot

    def field_of(n) -> str | None:
        parent = n.parent
        if parent is None:
            return None
        for i in range(parent.child_count):
            if parent.child(i) == n:
                return parent.field_name_for_child(i)
        return None

    def is_called(n) -> bool:
        """Is this attribute access the function of a call?

        `req.email` and `data.address` are DATA fields and must be renamed, or
        two copies of the same validation block stop matching just because one
        struct spells the field differently. `.strip()` and `.match()` are
        METHOD names and must be kept, or a read and a delete collapse into one
        clone. The grammar separates them: a method's attribute node is the
        `function` field of an enclosing call.
        """
        attr = n.parent
        if attr is None:
            return False
        return field_of(attr) == "function"

    def is_attribute(n) -> bool:
        if n.type in ATTRIBUTE_TYPES:
            return is_called(n)
        return field_of(n) in ATTRIBUTE_FIELDS and is_called(n)

    def skip(n) -> bool:
        return n.type in COMMENT_TYPES or bool(getattr(n, "is_extra", False))

    def emit_leaf(n) -> None:
        if not n.is_named:
            if n.type not in IGNORED_ANONYMOUS:
                out.append("@" + n.type)
            return
        if _is_literal(n.type):
            # The type alone. A changed constant is the commonest Type-2 edit
            # and must not split a clone class.
            out.append(n.type)
        elif n.type in IDENTIFIER_TYPES and not is_attribute(n):
            out.append(placeholder(_node_text(n, source)))
        else:
            out.append(f"{n.type}:{_node_text(n, source)}")

    def walk(n) -> None:
        if skip(n):
            return
        if not n.children:
            emit_leaf(n)
            return
        out.append("(" + n.type)
        for child in n.children:
            if skip(child):
                continue
            if child.is_named:
                walk(child)
            elif child.type not in IGNORED_ANONYMOUS:
                out.append("@" + child.type)
        out.append(")")

    walk(node)
    return out


def normalize(node, source: bytes, keep_identifiers: bool,
              include_operators: bool = True) -> list[str]:
    """Flatten a subtree into a normalised token sequence.

    Named children carry the structure. Meaningful anonymous tokens - the
    operators and keywords the grammar spells as literals - are emitted with an
    `@` prefix so they cannot be confused with a node type. Comments and extras
    are dropped explicitly: a note about why code exists should not be
    invalidated by someone rewording a docstring.

    The `@` prefix matters. Without it an anonymous `in` token and a node type
    called `in` would collide in the token stream, and a fingerprint is only
    worth anything if two different programs cannot produce the same one.

    `include_operators=False` reproduces the pre-v5 behaviour exactly, where
    only named children were walked. It exists for one purpose: proving, during
    a migration, that a fingerprint changed because this function changed and
    not because the code did. Nothing else may use it - it is the algorithm
    that let `a + b` and `a - b` hash identically.
    """
    out: list[str] = []

    def skip(n) -> bool:
        return n.type in COMMENT_TYPES or bool(getattr(n, "is_extra", False))

    def walk(n) -> None:
        if skip(n):
            return
        children = n.children if include_operators else n.named_children
        if not children:
            if n.is_named:
                out.append(f"{n.type}:{_node_text(n, source)}"
                           if keep_identifiers else n.type)
            elif include_operators and n.type not in IGNORED_ANONYMOUS:
                out.append("@" + n.type)
            return
        out.append("(" + n.type)
        for child in children:
            if skip(child):
                continue
            if child.is_named:
                walk(child)
            elif include_operators and child.type not in IGNORED_ANONYMOUS:
                out.append("@" + child.type)
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
    decorators: list[str] = field(default_factory=list)
    # receiver name -> declared or constructed type names. Keys prefixed with
    # `self.` are fields of the enclosing class rather than locals; see
    # _collect_receiver_types.
    receiver_types: dict[str, list[str]] = field(default_factory=dict)


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
                            "constant", "property_identifier", "word",
                            # Kotlin names every declaration with a
                            # `simple_identifier` and has no `name` field, so
                            # without this the search fell through the name and
                            # returned the RETURN TYPE: every Kotlin function
                            # was stored as `String`.
                            "simple_identifier"):
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


DECORATOR_NODES = {"decorator", "annotation", "marker_annotation",
                   "attribute", "attribute_list"}

# Nodes that group a symbol's annotations in with its access modifiers. Java
# and Kotlin both put `@Test` and `public` inside one `modifiers` node, so a
# scan of direct children finds the modifiers node and nothing inside it.
DECORATOR_CONTAINERS = {"modifiers", "modifier_list", "modifier"}

# Attributes written as a preceding sibling rather than a child. Rust puts
# `#[test]` before the function it applies to, at the same level.
SIBLING_DECORATOR_NODES = {"attribute_item"}


def _decorators_of(child, parent, source: bytes) -> list[str]:
    """Decorators, annotations and attributes attached to a symbol.

    Three shapes, and all three are load-bearing: an annotation is what makes a
    route or a test recognisable, so a shape that is not read does not merely
    lose a label - it empties the entry-point table for that language, silently.

        wrapper   Python hangs decorators off a `decorated_definition` node
                  around the function, so the parent has to be consulted.
        grouped   Java and Kotlin fold annotations in with `public` and `final`
                  inside a `modifiers` node, so a scan of direct children sees
                  only the group. This is why `@Test` and `@GetMapping` were
                  extracted from no Java file at all.
        sibling   Rust writes `#[test]` as an `attribute_item` BEFORE the
                  function, not inside it.
    """
    found: list[str] = []
    if parent is not None and parent.type == "decorated_definition":
        for node in parent.named_children:
            if node.type in DECORATOR_NODES:
                found.append(_node_text(node, source).strip()[:160])

    # Preceding siblings run backwards, so they are collected and reversed:
    # decorators are reported in source order in every other language and a
    # reader comparing two of them should not have to know which is which.
    preceding: list[str] = []
    sibling = child.prev_named_sibling
    while sibling is not None and sibling.type in SIBLING_DECORATOR_NODES:
        preceding.append(_node_text(sibling, source).strip()[:160])
        sibling = sibling.prev_named_sibling
    found.extend(reversed(preceding))

    for node in child.named_children:
        if node.type in DECORATOR_NODES:
            found.append(_node_text(node, source).strip()[:160])
        elif node.type in DECORATOR_CONTAINERS:
            for inner in node.named_children:
                if inner.type in DECORATOR_NODES:
                    found.append(_node_text(inner, source).strip()[:160])
    return found[:12]


def _semantic_children(node) -> list:
    """Named children that carry meaning: no comments, no extras.

    Anchoring reads structure, and a comment is not structure. Counting one as
    a sibling let a note added above a function change that function's
    ast_path and its neighbours' context fingerprints - moving an anchor for a
    change fingerprints deliberately ignore.
    """
    return [child for child in node.named_children
            if child.type not in COMMENT_TYPES
            and not getattr(child, "is_extra", False)]


def _is_function_binding(node) -> bool:
    """Whether a `variable_declarator` binds a function rather than a value."""
    value = node.child_by_field_name("value")
    return value is not None and value.type in VALUE_FUNCTION_NODES


def _declared_owner(node, source: bytes, lang: str) -> list[str]:
    """Container segments a symbol names in its own syntax rather than by nesting.

    Two languages declare a member outside the thing it belongs to, and in both
    the owner is written at the declaration instead of implied by nesting:

        Go   `func (c *Coordinator) Acquire()` is a sibling of every free
             function in the file. Stored as plain `Acquire` it cannot be told
             apart from a method of that name on another type, so every Go
             method call resolved to two or more candidates and was dropped.
        C++  `void app::C::m() {}` defines a method whose class is named only
             in the declarator, so the definition was stored under the
             enclosing namespace and never joined to its class at all.
    """
    if lang == "go":
        receiver = node.child_by_field_name("receiver")
        if receiver is None:
            return []
        stack = list(receiver.named_children)
        while stack:
            current = stack.pop(0)
            if current.type == "type_identifier":
                text = _node_text(current, source).strip()
                return [text] if text else []
            stack.extend(current.named_children)
        return []

    if lang in ("cpp", "c"):
        declarator = node.child_by_field_name("declarator")
        # `void app::C::m()` nests right: scope `app`, name `C::m`, and so on.
        while declarator is not None and declarator.type == "function_declarator":
            declarator = declarator.child_by_field_name("declarator")
        scopes: list[str] = []
        while declarator is not None and declarator.type == "qualified_identifier":
            scope = declarator.child_by_field_name("scope")
            if scope is not None:
                text = _node_text(scope, source).strip()
                if text:
                    scopes.append(text)
            declarator = declarator.child_by_field_name("name")
        return scopes

    return []


def _without_repeated_scope(owner: list[str], container_path: list[str]) -> list[str]:
    """Drop owner segments the enclosing containers already supply.

    `namespace app { void app::C::m() {} }` is legal and names `app` twice, but
    `app.app.C.m` is not a path any caller would write.
    """
    for overlap in range(min(len(owner), len(container_path)), 0, -1):
        if container_path[-overlap:] == owner[:overlap]:
            return owner[overlap:]
    return owner


def _impl_identity(node, source: bytes) -> tuple[str, str] | None:
    """(display name, container segment) for a Rust `impl` block.

    `struct S` and `impl S` are different nodes that both resolve to the name
    `S`, so storing both under `S` left two active symbols sharing one
    symbol_path - and call resolution refuses to choose between two candidates,
    so every call naming `S` became permanently ambiguous. The block is shown
    as `impl S` (or `impl Display for S`) while the methods inside it stay
    under `S`, which is what a caller actually writes.
    """
    type_node = node.child_by_field_name("type")
    if type_node is None:
        return None
    owner = _node_text(type_node, source).strip()
    if not owner:
        return None
    trait_node = node.child_by_field_name("trait")
    if trait_node is not None:
        label = f"impl {_node_text(trait_node, source).strip()} for {owner}"
    else:
        label = f"impl {owner}"
    return label[:120], owner


def _collect_calls(node, source: bytes, limit: int = 200) -> list[str]:
    """Callee names this symbol itself calls, in source order.

    Scope-aware, and both halves of that matter.

    The walk stops at a nested callable. Descending into one attributed the
    inner function's calls to the outer, so `def outer(): ...` containing a
    `def inner(): hidden()` produced a CALLS edge from `outer` to `hidden` -
    an edge for a call `outer` does not make. The same fault let a JS class
    absorb the calls of its arrow-function fields, which is worse than an
    incomplete graph: it points a real edge at the wrong symbol.

    It does NOT stop at a callable that IS this symbol's own body. `const f =
    () => g()` is a symbol whose value is an arrow function, so refusing to
    enter the arrow would leave it with no calls at all.

    Order is source order. The traversal is LIFO, so children are pushed
    reversed; without that `first(); second()` came back as `["second",
    "first"]`, and anything reading calls_raw positionally saw the body
    backwards.
    """
    found: list[str] = []

    def push(stack, current) -> None:
        stack.extend(reversed(current.named_children))

    # A symbol bound to a function value - `const f = () => g()` - keeps its
    # body one level down. Start inside that body, or the very first node
    # popped would be a nested callable and the symbol would have no calls.
    body = node
    if node.type in BINDING_NODES:
        value = node.child_by_field_name("value")
        if value is not None and value.type in VALUE_FUNCTION_NODES:
            body = value

    stack: list[Any] = []
    push(stack, body)

    while stack and len(found) < limit:
        current = stack.pop()
        if current.type in CALL_NODES:
            target = (current.child_by_field_name("function")
                      or current.child_by_field_name("name"))
            if target is not None:
                text = _node_text(target, source).strip()
                if text:
                    found.append(text.split("(")[0].strip()[:120])
            # Arguments can hold calls this symbol also makes - f(g()) - so a
            # call node is never a stopping point, only the callee is recorded.
            push(stack, current)
            continue
        if current.type in FUNCTION_BODY_NODES:
            continue                    # a nested callable owns its own calls
        push(stack, current)
    return found


# Nodes that bind a name to a value one statement at a time. Their shape is
# the same across the languages this extracts from: a left-hand target and a
# right-hand value, or a declarator carrying both.
TYPE_BINDING_NODES = {
    "assignment", "assignment_expression", "variable_declarator",
    "short_var_declaration", "let_declaration", "const_declaration",
    "var_declaration", "field_definition", "public_field_definition",
    "local_variable_declaration", "parameter", "typed_parameter",
    "formal_parameter", "required_parameter", "optional_parameter",
    "var_spec",
}

# Nodes whose text names a constructed type: `new Foo()` in JS, Java and C#.
# Python and Rust construct with a plain call instead.
CONSTRUCTION_NODES = {"new_expression", "object_creation_expression"}


def _type_name(text: str) -> str:
    """The bare type a declaration or construction names, or "".

    Generics, pointers, call parentheses and module qualification are all
    stripped, because resolution looks the result up as a container name: a
    receiver declared `*store.Cache` and one declared `Cache<K>` both have to
    reach the container indexed as `Cache`.
    """
    text = text.strip().split("\n")[0].strip()
    for cut in ("(", "<", "[", "{"):
        text = text.split(cut)[0]
    text = text.strip().lstrip("*&").strip()
    # `store.Cache` / `store::Cache` names the type in its last segment.
    for sep in ("::", "."):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    return text if text.isidentifier() else ""


def _value_type(node, source: bytes) -> str:
    """The type a value expression constructs, or "".

    Deliberately shallow: only a direct construction counts. Following a value
    through a helper would need a real dataflow pass, and a wrong answer here
    resolves a call to the wrong method rather than leaving it unresolved.
    """
    if node is None:
        return ""
    if node.type in CONSTRUCTION_NODES:
        target = (node.child_by_field_name("type")
                  or node.child_by_field_name("constructor"))
        if target is not None:
            return _type_name(_node_text(target, source))
        return ""
    if node.type in CALL_NODES:
        target = (node.child_by_field_name("function")
                  or node.child_by_field_name("name"))
        if target is not None:
            return _type_name(_node_text(target, source))
    return ""


def _binding_targets(node, source: bytes) -> list[str]:
    """Receiver keys a binding declares. `self.x` and `this.x` keep the prefix,
    which is what tells a field apart from a local at resolution time."""
    target = (node.child_by_field_name("left")
              or node.child_by_field_name("name")
              or node.child_by_field_name("property")
              or node.child_by_field_name("declarator")
              or node.child_by_field_name("pattern"))
    if target is None:
        return []
    text = _node_text(target, source).strip().split("\n")[0].strip()
    if not text:
        return []
    if text.isidentifier():
        return [text]
    head, _, tail = text.partition(".")
    if head in ("self", "this") and tail.isidentifier():
        return ["self." + tail]
    return []


def _collect_receiver_types(node, source: bytes,
                            limit: int = 120) -> dict[str, list[str]]:
    """Receiver name -> the types it may hold, within this symbol.

    This is the evidence the receiver_typed call tier runs on. It is a record
    of type flow, not a type system: a name that gets two bindings keeps both,
    and resolution refuses to answer unless exactly one of them yields a
    method. Recording only the last binding would be a strong update, which is
    unsound the moment a name is reassigned on a branch or aliased.

    Scoping matches _collect_calls: the walk stops at a nested callable, so an
    inner function's locals never masquerade as the outer symbol's.
    """
    found: dict[str, list[str]] = {}

    def add(name: str, type_name: str) -> None:
        if not name or not type_name or len(found) >= limit:
            return
        seen = found.setdefault(name, [])
        if type_name not in seen:
            seen.append(type_name)

    def push(stack, current) -> None:
        stack.extend(reversed(current.named_children))

    stack: list[Any] = []
    push(stack, node)
    while stack:
        current = stack.pop()
        if current.type in FUNCTION_BODY_NODES:
            continue                    # a nested callable owns its own locals
        if current.type in TYPE_BINDING_NODES:
            declared = current.child_by_field_name("type")
            value = (current.child_by_field_name("value")
                     or current.child_by_field_name("right"))
            type_name = ""
            if declared is not None:
                type_name = _type_name(_node_text(declared, source))
            if not type_name:
                type_name = _value_type(value, source)
            if type_name:
                for target in _binding_targets(current, source):
                    add(target, type_name)
        push(stack, current)
    return found


def _module_symbol(tree, source: bytes, lang: str,
                   symbols: list[ParsedSymbol]) -> "ParsedSymbol | None":
    """A pseudo-symbol owning the calls made at file scope, or None.

    Without it those calls have no caller, so no CALLS edge is written and
    their callee looks unreferenced - the known false positive behind
    graph(action='deadcode') reporting a function that module-level code
    plainly invokes. It is named `<module>` rather than after the file so it
    can never be mistaken for a declaration: no callee name can match it, and
    its `module` kind keeps it out of the dead-code listing.
    """
    root = tree.root_node
    spans = [(sym.start_byte, sym.end_byte) for sym in symbols]

    def inside_symbol(current) -> bool:
        return any(start <= current.start_byte and current.end_byte <= end
                   for start, end in spans)

    calls: list[str] = []
    stack: list[Any] = list(reversed(root.named_children))
    while stack and len(calls) < 200:
        current = stack.pop()
        if current.type in FUNCTION_BODY_NODES or inside_symbol(current):
            continue
        if current.type in CALL_NODES:
            target = (current.child_by_field_name("function")
                      or current.child_by_field_name("name"))
            if target is not None:
                text = _node_text(target, source).strip()
                if text:
                    calls.append(text.split("(")[0].strip()[:120])
        stack.extend(reversed(current.named_children))

    if not calls:
        return None
    # Fingerprinted over the calls themselves rather than the file: carrying
    # them is the only reason this row exists, so it should be re-resolved
    # when they change and left alone when they do not.
    digest = _sha("\n".join(calls))
    return ParsedSymbol(
        name="<module>",
        symbol_path="<module>",
        kind="module",
        lang=lang,
        signature="<module scope>",
        ast_path="module",
        start_byte=0,
        end_byte=root.end_byte,
        line_start=1,
        line_end=root.end_point[0] + 1,
        content_fingerprint=digest,
        skeleton_fingerprint=digest,
        token_signature=token_signature(calls),
        calls=calls,
    )


def _symbols_from_tags(tree, source: bytes, lang: str) -> list[ParsedSymbol]:
    """Symbols for a language with no hand-written rules, from its tags.scm.

    This is what makes coverage wide without making it dishonest. The hand
    maps stay authoritative for the languages they cover, because they know
    things a generic query cannot - a Go method's receiver, a Rust impl block,
    a C++ definition written outside its class. Everything else gets real
    symbols and real calls instead of an empty list.

    Nesting is recovered by byte containment rather than from the query, since
    tags.scm describes definitions individually and says nothing about which
    encloses which.
    """
    query = _tags_query(lang)
    if query is None:
        return []
    try:
        import tree_sitter
        cursor = tree_sitter.QueryCursor(query)
        matches = cursor.matches(tree.root_node)
    except Exception:
        return []

    found: list[tuple[Any, str, str]] = []      # (node, kind, name)
    calls: list[tuple[int, int, str]] = []      # (start, end, callee name)
    for _, captures in matches:
        name_nodes = captures.get("name") or []
        if not name_nodes:
            continue
        label = _node_text(name_nodes[0], source).strip().split("\n")[0]
        for capture, nodes in captures.items():
            if capture == "reference.call":
                # `@reference.call` marks the whole invocation and `@name` the
                # callee within it. Taking the invocation's own text recorded
                # an Elixir call as "defmodule Auth do\n  def refresh".
                for node in nodes:
                    if label:
                        calls.append((node.start_byte, node.end_byte, label[:120]))
                continue
            kind = TAG_KINDS.get(capture)
            if not kind:
                continue
            for node in nodes:
                if label:
                    found.append((node, kind, label))

    # Outermost first, so a container is always seen before what it holds.
    # A definition that IS a call drops out of the call list. Elixir declares
    # with macros, so `defmodule Auth do ... end` is captured both as a module
    # definition and as a call to `defmodule` over the same bytes - recording
    # the second would put `defmodule` and `def` in the call graph of every
    # Elixir file.
    definition_spans = {(node.start_byte, node.end_byte) for node, _, _ in found}
    calls = [call for call in calls if (call[0], call[1]) not in definition_spans]

    found.sort(key=lambda item: (item[0].start_byte, -item[0].end_byte))
    calls.sort()

    symbols: list[ParsedSymbol] = []
    stack: list[tuple[Any, str, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for node, kind, name in found:
        key = (node.start_byte, node.end_byte, name)
        if key in seen:
            continue
        seen.add(key)
        while stack and node.start_byte >= stack[-1][0].end_byte:
            stack.pop()
        container = [entry[2] for entry in stack if entry[1] in CONTAINER_KINDS]
        symbol_path = ".".join(container + [name])

        content_tokens = normalize(node, source, keep_identifiers=True)
        own_calls = [
            callee for start, end, callee in calls
            if node.start_byte <= start and end <= node.end_byte and callee != name
        ]
        symbols.append(ParsedSymbol(
            name=name,
            symbol_path=symbol_path,
            kind=kind,
            lang=lang,
            signature=_signature(node, source, name),
            ast_path=f"tags:{kind}",
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            content_fingerprint=_sha("".join(content_tokens)),
            skeleton_fingerprint=_sha("".join(
                normalize(node, source, keep_identifiers=False))),
            token_signature=token_signature(content_tokens),
            body=_node_text(node, source)[:4000],
            calls=own_calls[:200],
            receiver_types=_collect_receiver_types(node, source),
        ))
        if kind in CONTAINER_KINDS:
            stack.append((node, kind, name))

    module = _module_symbol(tree, source, lang, symbols)
    if module is not None:
        symbols.append(module)
    return symbols


# Minimum size, in NORMALISED tokens, for a fragment to be worth storing.
#
# Not PMD CPD's 50-100. That guidance is in RAW lexical tokens and does not
# transfer: normalize() emits structural markers (`(node_type`, `@operator`)
# that inflate the count far above a lexer's. Measured over 2787 fragments in
# src/icn, p50=91 and p90=405, and floors of 20 and 30 filter NOTHING - they
# return identical counts. Real discrimination happens between 40 and 100:
#
#     floor 20 -> 2787      floor  60 -> 1863
#     floor 30 -> 2787      floor 100 -> 1319
#
# 60 is the chosen start: it removes a third of the rows while keeping
# fragments as small as the four-line validation block clone detection exists
# to catch. Applied BEFORE insert, never at query time.
MIN_FRAGMENT_TOKENS = 60

# Bodies worth extracting as fragments in their own right. A clone is very
# often a loop body or a guard block repeated inside two otherwise unrelated
# functions, which whole-symbol fingerprints structurally cannot see: visit()
# computes fingerprints only for declaration nodes and never decomposes a
# function body. NiCad supports block granularity for exactly this reason.
_FRAGMENT_NODES = {
    "block", "statement_block", "compound_statement",
    "if_statement", "else_clause", "elif_clause",
    "for_statement", "while_statement", "do_statement",
    "foreach_statement", "for_in_statement", "for_range_loop",
    "try_statement", "catch_clause", "except_clause", "finally_clause",
    "with_statement", "switch_statement", "match_statement",
    "case_statement", "match_arm", "switch_case", "when_entry",
}


@dataclass
class ParsedFragment:
    """A syntactically meaningful region worth comparing against other code.

    Deliberately not a symbol: it has no name, no identity across edits, and
    no anchor. Fragments are derived data, rebuilt from scratch whenever their
    file is re-stored, so nothing is lost by deleting and re-inserting them.
    """
    symbol_path: str
    lang: str
    kind: str
    start_byte: int
    end_byte: int
    line_start: int
    line_end: int
    token_count: int
    content_fingerprint: str
    alpha_fingerprint: str
    skeleton_fingerprint: str
    token_signature: str


def extract_fragments(source: bytes, lang: str,
                      min_tokens: int = MIN_FRAGMENT_TOKENS) -> list[ParsedFragment]:
    """Block-level clone candidates for one file.

    Separate from parse_file rather than folded into it, because parse_file's
    output feeds anchoring: every fingerprint it produces is load-bearing for
    memories already stored against this code. Keeping fragment extraction on
    its own path means the symbol fingerprints stay byte-identical and no
    existing anchor moves.

    Regions are nested by nature - a function body contains an if body contains
    a loop body - and every level is a legitimate clone candidate at a
    different granularity, so nesting is kept. Clustering, not extraction, is
    where overlapping matches get merged into their maximal region.
    """
    parser = get_parser(lang)
    if parser is None:
        return []
    try:
        tree = parser.parse(source)
    except Exception:
        return []

    fragments: list[ParsedFragment] = []
    seen: set[tuple[int, int]] = set()

    def walk(node, enclosing: str) -> None:
        for child in _semantic_children(node):
            name = _name_of(child, source)
            owner = enclosing
            if child.type in SYMBOL_NODES.get(lang, {}) and name:
                owner = f"{enclosing}.{name}" if enclosing else name
                _emit(child, owner, child.type)
            elif child.type in _FRAGMENT_NODES:
                _emit(child, owner, child.type)
            walk(child, owner)

    def _emit(node, owner: str, kind: str) -> None:
        # A body that is its parent's only child spans the same bytes as the
        # parent, so storing both would report a fragment as a clone of itself.
        span = (node.start_byte, node.end_byte)
        if span in seen:
            return
        content = normalize(node, source, keep_identifiers=True)
        if len(content) < min_tokens:
            return
        seen.add(span)
        fragments.append(ParsedFragment(
            symbol_path=owner,
            lang=lang,
            kind=kind,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            token_count=len(content),
            content_fingerprint=_sha("".join(content)),
            alpha_fingerprint=_sha("".join(alpha_normalize(node, source))),
            skeleton_fingerprint=_sha("".join(
                normalize(node, source, keep_identifiers=False))),
            token_signature=token_signature(content),
        ))

    walk(tree.root_node, "")
    return fragments


def parse_file(path: Path, source: bytes, lang: str) -> list[ParsedSymbol]:
    """Extract symbols from one file. Returns [] for anything unparseable."""
    parser = get_parser(lang)
    if parser is None:
        return []
    try:
        tree = parser.parse(source)
    except Exception:
        return []
    node_map = SYMBOL_NODES.get(lang, {})
    if not node_map:
        return _symbols_from_tags(tree, source, lang)

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
        siblings = _semantic_children(node)
        # Ordinal among siblings OF THE SAME TYPE, not among all of them.
        # A raw positional index made ast_path depend on comments - the one
        # thing fingerprints go out of their way to ignore - so adding a note
        # above a function renumbered it and moved its anchor.
        seen_of_type: dict[str, int] = {}
        for index, child in enumerate(siblings):
            ordinal = seen_of_type.get(child.type, 0)
            seen_of_type[child.type] = ordinal + 1
            child_ast = ast_path + [f"{child.type}[{ordinal}]"]
            kind = node_map.get(child.type)
            if kind is None:
                visit(child, container_path, child_ast)
                continue

            if child.type == "variable_declarator" and not _is_function_binding(child):
                visit(child, container_path, child_ast)
                continue

            name = _name_of(child, source)
            # A Rust impl block reports itself under a different label than the
            # container it opens, so the two are tracked separately from here.
            container_segment = name
            if child.type == "impl_item":
                identity = _impl_identity(child, source)
                if identity:
                    name, container_segment = identity
            if not name:
                visit(child, container_path, child_ast)
                continue

            owner = _without_repeated_scope(
                _declared_owner(child, source, lang), container_path)
            owned_path = container_path + owner
            symbol_path = ".".join(owned_path + [name])
            content_tokens = normalize(child, source, keep_identifiers=True)
            skeleton_tokens = normalize(child, source, keep_identifiers=False)
            own_fingerprint = _sha("".join(content_tokens))
            fingerprint_cache[child.id] = own_fingerprint

            # Context fingerprints use semantic siblings for the same reason
            # ast_path uses a same-type ordinal: a comment inserted between two
            # functions normalises to nothing, so counting it as the neighbour
            # replaced a real context hash with the hash of an empty string.
            prev_fp = next_fp = None
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
                decorators=_decorators_of(child, node, source),
                receiver_types=_collect_receiver_types(child, source),
            ))

            next_container = (owned_path + [container_segment]
                              if kind in CONTAINER_KINDS else container_path)
            visit(child, next_container, child_ast)

    visit(tree.root_node, [], [])
    module = _module_symbol(tree, source, lang, symbols)
    if module is not None:
        symbols.append(module)
    return symbols


# Nodes that carry a module specifier inside an import statement. Strings are
# unwrapped; dotted/scoped names are taken whole rather than per-identifier.
_MODULE_LITERALS = {
    "string", "interpreted_string_literal", "string_literal", "system_lib_string",
    "raw_string_literal", "string_fragment",
}
_MODULE_NAMES = {"dotted_name", "scoped_identifier", "relative_import", "identifier",
                 # C# `using App.Storage;` hangs a `qualified_name` under the
                 # directive. Without it the sweep recursed to the identifiers
                 # inside and reported two modules, `App` and `Storage`, neither
                 # of which was ever imported.
                 "qualified_name", "qualified_identifier", "namespace_use_clause"}

_QUOTES = "\"'`<>"


def _module_text(node, source: bytes) -> str:
    return _node_text(node, source).strip().strip(_QUOTES).strip()


def _python_import_modules(node, source: bytes) -> list[tuple[str, int, bool]]:
    """(module, relative_level, is_probe) for one Python import node.

    Python needs the field-specific path rather than the generic descendant
    sweep: in `from a.b import c, d` the imported names are `dotted_name`
    nodes too, so sweeping descendants would report `c` and `d` as modules.

    But `from a import b` genuinely does import the submodule `a.b` when one
    exists, and that is how this codebase imports its own modules
    (`from . import parsing`). Ignoring it linked every such file to the
    package `__init__` instead of the module actually used. So each imported
    name is also emitted as a *probe*: resolved if a matching file exists, and
    otherwise dropped silently rather than recorded as an external module -
    a probe that misses is usually just a class or function name.
    """
    out: list[tuple[str, int, bool]] = []
    if node.type == "import_from_statement":
        module_node = node.child_by_field_name("module_name")
        if module_node is None:
            return out
        text = _module_text(module_node, source)
        level = len(text) - len(text.lstrip("."))
        base = text[level:]
        out.append((base, level, False))
        for name_node in node.children_by_field_name("name"):
            target = name_node
            if target.type == "aliased_import":
                target = target.child_by_field_name("name") or target
            name = _module_text(target, source)
            if name and name != "*":
                out.append((f"{base}.{name}" if base else name, level, True))
        return out

    # Plain `import a.b, c as d` - modules are the direct dotted children.
    for child in node.named_children:
        target = child
        if child.type == "aliased_import":
            target = child.child_by_field_name("name") or child
        if target.type in ("dotted_name", "identifier"):
            text = _module_text(target, source)
            if text:
                out.append((text, 0, False))
    return out


def _js_import_modules(node, source: bytes) -> list[tuple[str, int, bool]]:
    """Module specifier for one JS/TS `import` or `export ... from` statement.

    Read off the `source` field rather than swept out of the subtree. The sweep
    took the imported NAMES as modules too, so `import { hashToken } from
    "./crypto"` recorded a dependency on a module called `hashToken` - which
    has never existed anywhere - alongside the real one. Every named import in
    the repository produced one of those, and they are indistinguishable from
    genuine third-party dependencies once written as `extern:`.
    """
    module_node = node.child_by_field_name("source")
    if module_node is None:
        return []
    text = _module_text(module_node, source)
    return [(text, 0, False)] if text else []


def _require_module(node, source: bytes) -> tuple[str, bool] | None:
    """(module, always_deferred) for a `require("x")` / `import("x")` call.

    CommonJS states a dependency with an ordinary function call, so no grammar
    marks it as an import and the statement-node scan cannot see it at all: a
    `require`-based JavaScript file had no imports whatsoever, and neither did
    a Ruby file using `require_relative`.

    A dynamic `import()` is deferred wherever it appears. It returns a promise
    and loads at await time, so unlike `require` it cannot force an
    initialisation order even at module top level - which is exactly what makes
    it the standard way to break a JavaScript import cycle.
    """
    function_node = node.child_by_field_name("function")
    if function_node is None:
        return None
    callee = _node_text(function_node, source).strip()
    if callee not in REQUIRE_CALLS:
        return None
    arguments = node.child_by_field_name("arguments")
    if arguments is None:
        return None
    for argument in arguments.named_children:
        if argument.type in _MODULE_LITERALS:
            text = _module_text(argument, source)
            if text:
                return text, callee == "import"
        # A computed specifier - require(base + name) - is a real dependency
        # that cannot be named. Reporting the first string inside it would name
        # a fragment, so nothing is reported.
        return None
    return None


def _is_type_checking_guard(node, source: bytes) -> bool:
    """Whether this node is an `if TYPE_CHECKING:` block.

    An import under one never runs, so it cannot close an import cycle - it is
    Python's equivalent of a TypeScript `import type`, and is the standard way
    to type-annotate across a cycle without creating one.

    Deliberately only this condition, matched exactly. Any other conditional
    import is a real runtime import: `if platform.system() == "Windows": import
    winreg` genuinely executes during module initialisation, and treating every
    guarded import as deferred would hide real cycles.
    """
    if node.type != "if_statement":
        return False
    condition = node.child_by_field_name("condition")
    if condition is None:
        return False
    text = _node_text(condition, source).strip()
    return text in ("TYPE_CHECKING", "typing.TYPE_CHECKING", "t.TYPE_CHECKING")


def _go_import_modules(node, source: bytes) -> list[tuple[str, int, bool]]:
    """Module paths for a Go import block, read from each spec's `path` field.

    `import foo "example.com/x"` names the module in `path` and the alias in
    `name`. The generic sweep visits both children, and the only reason it did
    not report `foo` as a module is that Go spells an alias `package_identifier`
    rather than `identifier` - a coincidence of grammar naming, not a rule.
    Reading the field makes it a rule.
    """
    out: list[tuple[str, int, bool]] = []
    stack = [node]
    while stack and len(out) < 64:
        current = stack.pop()
        if current.type == "import_spec":
            path = current.child_by_field_name("path")
            if path is not None:
                text = _module_text(path, source)
                if text:
                    out.append((text, 0, False))
            continue
        stack.extend(reversed(current.named_children))
    return out


def _rust_module_specs(node, source: bytes) -> list[tuple[str, int, bool]] | None:
    """`mod name;` as an import. Returns None if this node is not one.

    A Rust module declaration is how a crate states that another FILE is part
    of it, which makes it the only edge that reveals the crate's file tree -
    `use` paths alone cannot, because they go through the module namespace.
    An inline `mod name { .. }` declares no file and is rejected here; it still
    has to be walked, so it cannot be rejected by the node type alone.
    """
    if node.child_by_field_name("body") is not None:
        return None
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    text = _module_text(name_node, source)
    return [(text, 0, False)] if text else None


def _generic_import_modules(node, source: bytes) -> list[tuple[str, int, bool]]:
    """Module specifiers for languages with no dedicated extractor.

    Takes the outermost literal or dotted name under the import node and does
    not recurse into it, so `com.example.Foo` stays one module rather than
    three identifiers.
    """
    out: list[tuple[str, int, bool]] = []
    stack = list(node.named_children)
    while stack and len(out) < 8:
        current = stack.pop(0)
        if current.type in _MODULE_LITERALS or current.type in _MODULE_NAMES:
            text = _module_text(current, source)
            if text:
                out.append((text, 0, False))
            continue
        stack.extend(current.named_children)
    return out


def file_import_specs(source: bytes, lang: str) -> list[dict[str, Any]]:
    """Module specifiers imported by this file, as structured records.

    Returns the specifier as written (`./auth`, `icn.db`, `com.example.Foo`)
    rather than the statement text. Resolution to a real file happens in a
    second pass - see `imports.resolve_file_imports` - because during a walk a
    file is routinely indexed before the file it imports.
    """
    parser = get_parser(lang)
    if parser is None:
        return []
    try:
        tree = parser.parse(source)
    except Exception:
        return []

    # Keyed by (module, level) so one module imported twice yields one spec.
    #
    # A dict rather than a set of seen keys, because the two occurrences may
    # disagree about deferredness and the walk is LIFO - so source order is not
    # visit order, and a plain first-wins dedup would let a function-level
    # import at the bottom of a file mask the module-level one at the top.
    # Eager always wins: a module imported both ways is eagerly imported.
    found: dict[tuple[str, int], dict[str, Any]] = {}
    # Ancestry matters here, so the walk carries whether we are inside a
    # function body. A deferred import is a real dependency but not an
    # initialisation-order one, and cycle detection has to tell them apart.
    # Ancestry matters, so the walk carries two independent reasons an import
    # below this point cannot force an initialisation order.
    stack: list[tuple[Any, bool, bool]] = [(tree.root_node, False, False)]
    while stack and len(found) < 300:
        node, inside_function, type_only = stack.pop()

        modules: list[tuple[str, int, bool]] | None = None
        always_deferred = False
        descend = True
        if node.type in IMPORT_NODES:
            if node.type == "mod_item":
                modules = _rust_module_specs(node, source)
            elif lang == "python":
                modules = _python_import_modules(node, source)
            elif lang in JS_IMPORT_LANGS:
                modules = _js_import_modules(node, source)
            elif lang == "go":
                modules = _go_import_modules(node, source)
            else:
                modules = _generic_import_modules(node, source)
            descend = node.type in IMPORT_NODES_WITH_BODIES
        elif node.type in EXPORT_IMPORT_NODES:
            # `export { X } from "./m"` and `export * from "./m"` are imports
            # wearing an export's clothes, and were producing no edge at all.
            # The node is NOT a leaf - `export function f() { import("./x") }`
            # is the same node type - so the subtree is still walked.
            modules = _js_import_modules(node, source)
        elif node.type in CALL_NODES:
            # A call is only ever a require, never a statement wrapping one, so
            # the subtree below it still has to be walked for nested calls.
            required = _require_module(node, source)
            if required:
                module, always_deferred = required
                modules = [(module, 0, False)]

        if modules:
            raw = _node_text(node, source).strip().replace("\n", " ")[:200]
            # A TypeScript `import type` is erased at compile time, so it can
            # no more force an initialisation order than a deferred import can.
            deferred = (inside_function or type_only or always_deferred
                        or raw.startswith("import type")
                        or raw.startswith("export type"))
            for module, level, probe in modules:
                if not module and not level:
                    continue
                key = (module, level)
                previous = found.get(key)
                if previous is not None and not previous.get("deferred"):
                    continue            # already recorded eagerly; eager wins
                spec = {"module": module, "level": level, "raw": raw}
                if probe:
                    spec["probe"] = True
                if deferred:
                    spec["deferred"] = True
                found[key] = spec

        if descend:
            nested = inside_function or node.type in FUNCTION_BODY_NODES
            typing_only = type_only or _is_type_checking_guard(node, source)
            stack.extend((child, nested, typing_only)
                         for child in node.named_children)
    return list(found.values())


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
