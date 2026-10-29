"""A small parser for the YAML subset used by experiment configs.

PyYAML is not available (the package is standard-library only), and the
experiment configs in this project use a deliberately small YAML subset:

- mappings nested by indentation (`key: value`, `key:` + indented block)
- lists, block style (`- item`) or flow style (`[a, b, c]`)
- scalars: plain strings, 'single' and "double" quoted strings,
  integers, floats, booleans (true/false), null (~, null, or empty)
- comments (`#` at line start, or after whitespace, outside quotes)

That subset is all a config needs: experiment name, question, provider,
model, grid axes, caps. Anything outside it -- anchors, multi-line
strings, lists of mappings, flow mappings -- raises YamlSubsetError with
a line number instead of being mis-parsed silently.
"""
from __future__ import annotations

import re


class YamlSubsetError(ValueError):
    pass


_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


def _strip_comment(line: str) -> str:
    out = []
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote == '"':
            out.append(ch)
            if ch == "\\" and i + 1 < len(line):
                out.append(line[i + 1])
                i += 2
                continue
            if ch == '"':
                quote = None
        elif quote == "'":
            out.append(ch)
            if ch == "'":
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _logical_lines(text: str):
    """(indent, content, lineno) for every non-blank, non-comment line."""
    lines = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        if "\t" in raw[:len(raw) - len(raw.lstrip())]:
            raise YamlSubsetError(f"line {lineno}: tabs are not supported "
                                  "for indentation")
        content = _strip_comment(raw)
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        lines.append((indent, content.strip(), lineno))
    return lines


def _split_flow(inner: str, lineno: int) -> list:
    """Split a flow-list body on top-level commas (quotes respected)."""
    parts, buf, quote = [], [], None
    for ch in inner:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf).strip())
            buf = []
        elif ch in "[]{}":
            raise YamlSubsetError(f"line {lineno}: nested flow collections "
                                  "are outside the supported subset")
        else:
            buf.append(ch)
    if quote:
        raise YamlSubsetError(f"line {lineno}: unterminated quoted string")
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p != ""]


def _parse_double_quoted(text: str, lineno: int) -> str:
    out = []
    i = 1
    escapes = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            out.append(escapes.get(text[i + 1], text[i + 1]))
            i += 2
            continue
        if ch == '"':
            if text[i + 1:].strip():
                raise YamlSubsetError(
                    f"line {lineno}: trailing characters after string")
            return "".join(out)
        out.append(ch)
        i += 1
    raise YamlSubsetError(f"line {lineno}: unterminated double-quoted string")


def parse_scalar(text: str, lineno: int = 0):
    """Parse one scalar (or flow list) from the supported subset."""
    text = text.strip()
    if text == "":
        return None
    if text.startswith("["):
        if not text.endswith("]"):
            raise YamlSubsetError(f"line {lineno}: unterminated flow list")
        return [parse_scalar(p, lineno) for p in _split_flow(text[1:-1], lineno)]
    if text.startswith("{"):
        raise YamlSubsetError(f"line {lineno}: flow mappings are outside "
                              "the supported subset")
    if text.startswith('"'):
        return _parse_double_quoted(text, lineno)
    if text.startswith("'"):
        if len(text) < 2 or not text.endswith("'"):
            raise YamlSubsetError(
                f"line {lineno}: unterminated single-quoted string")
        return text[1:-1].replace("''", "'")
    low = text.lower()
    if low in ("null", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if _INT.match(text):
        return int(text)
    if _FLOAT.match(text) and ("." in text or "e" in low):
        return float(text)
    return text


def _split_key(content: str, lineno: int):
    """Split a mapping line into (key, rest-after-colon)."""
    quote = None
    for i, ch in enumerate(content):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == ":" and (i + 1 == len(content) or content[i + 1] == " "):
            key = content[:i].strip()
            if not key:
                raise YamlSubsetError(f"line {lineno}: empty mapping key")
            if key.startswith(('"', "'")):
                key = parse_scalar(key, lineno)
            return key, content[i + 1:].strip()
    raise YamlSubsetError(f"line {lineno}: expected 'key: value', "
                          f"got {content!r}")


def _parse_block(lines, pos: int, indent: int):
    """Parse the block starting at lines[pos] (given indent); return
    (value, next_pos)."""
    if pos >= len(lines):
        return None, pos
    first_indent, first_content, _ = lines[pos]
    if first_indent < indent:
        return None, pos
    if first_content.startswith("- ") or first_content == "-":
        items = []
        while pos < len(lines):
            ind, content, lineno = lines[pos]
            if ind != indent or not (content.startswith("- ")
                                     or content == "-"):
                break
            item_text = content[1:].strip()
            pos += 1
            if item_text == "":
                # the item is a nested block on the following lines
                if pos < len(lines) and lines[pos][0] > indent:
                    val, pos = _parse_block(lines, pos, lines[pos][0])
                    items.append(val)
                else:
                    items.append(None)
            elif ":" in item_text and not item_text.startswith(("[", '"', "'")):
                raise YamlSubsetError(
                    f"line {lineno}: lists of mappings are outside the "
                    "supported subset")
            else:
                items.append(parse_scalar(item_text, lineno))
        return items, pos
    mapping = {}
    while pos < len(lines):
        ind, content, lineno = lines[pos]
        if ind != indent or content.startswith("- ") or content == "-":
            break
        key, rest = _split_key(content, lineno)
        if key in mapping:
            raise YamlSubsetError(f"line {lineno}: duplicate key {key!r}")
        pos += 1
        if rest != "":
            mapping[key] = parse_scalar(rest, lineno)
        elif pos < len(lines) and lines[pos][0] > indent:
            val, pos = _parse_block(lines, pos, lines[pos][0])
            mapping[key] = val
        else:
            mapping[key] = None
    return mapping, pos


def parse_yaml(text: str):
    """Parse the supported YAML subset into dicts/lists/scalars."""
    lines = _logical_lines(text)
    if not lines:
        return None
    value, pos = _parse_block(lines, 0, lines[0][0])
    if pos != len(lines):
        _, content, lineno = lines[pos]
        raise YamlSubsetError(f"line {lineno}: could not parse {content!r} "
                              "(inconsistent indentation?)")
    return value


def load_yaml(path):
    from pathlib import Path
    return parse_yaml(Path(path).read_text())
