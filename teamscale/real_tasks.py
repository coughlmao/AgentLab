"""Real codebase_repair tasks: actual broken Python projects on disk.

Where `tasks.make_codebase_repair` produces simulated payloads, this module
generates a *real* project per seed: 8 small Python modules in a sandbox
directory, each carrying exactly one seeded defect, plus one check script
per module. Ground truth is not a string comparison -- it is the check
script's exit code, obtained by really executing it in a subprocess
(strict: exit 0 or the subtask is wrong).

Freshness: module names, function names, constants, and test values are all
drawn from the seed, and the defect assigned to each slot is shuffled per
seed, so a fresh task instance cannot be solved from memory of a previous
one. A subtask = fix one module so its check passes.
"""
from __future__ import annotations

import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .tasks import ensure_within, write_in_sandbox

CHECK_TIMEOUT = 10  # seconds, per check execution

# Generator version for this task family (ENGINEERING.md, invariant 1
# and 3): recorded in every run manifest. Changing any template,
# name pool, or sampling rule below increments it.
GENERATOR_VERSION = "repair-gen-1"


@dataclass
class RealModule:
    """One subtask: a broken module and the check that exposes the break."""
    subtask_id: str
    module_file: str          # e.g. "pricing.py" (relative to sandbox root)
    check_file: str           # e.g. "check_pricing.py"
    func_name: str
    defect_type: str
    description: str          # what the agent is told (symptom level only)
    broken_code: str
    fixed_code: str
    check_code: str = ""
    # scripted repair for the offline fake backend: applying (find -> repl)
    # to broken_code yields fixed_code; verified at generation time.
    repair: tuple = ("", "")
    # hard mode only: extra sandbox files that ship with this subtask,
    # filename -> (broken_code, fixed_code). Used by cross-module defects,
    # where the caller in module_file is correct and the defect lives in a
    # helper it imports.
    extra_files: dict = field(default_factory=dict)
    # file the repair pair applies to; "" means module_file. Cross-module
    # defects are repaired in the helper, not the caller.
    repair_file: str = ""
    # how many distinct defects this subtask carries (multi-defect modules).
    n_defects: int = 1
    # hard mode only: (filename, code) variants that fix SOME but not all
    # defects; the check must still fail on each (interaction proof).
    partial_fixes: list = field(default_factory=list)


@dataclass
class RealTask:
    seed: int
    family: str = "codebase_repair"
    modules: list = field(default_factory=list)
    difficulty: str = "standard"

    def by_id(self):
        return {m.subtask_id: m for m in self.modules}


# ------------------------------------------------------------ defect templates
# Each builder takes spec {func, k} and returns
# (fixed_body_lines, broken_body_lines, cases, repair_pair), where cases is a
# list of (args_tuple, expected) exercised by the generated check script.

def _t_wrong_operator(spec):
    k = spec["k"]
    fixed = [f"return x + y * {k}"]
    broken = [f"return x - y * {k}"]
    cases = [((3, 2), 3 + 2 * k), ((0, 5), 5 * k), ((7, 1), 7 + k), ((2, 0), 2)]
    return fixed, broken, cases, (f"x - y * {k}", f"x + y * {k}")


def _t_off_by_one(spec):
    fixed = ["return list(range(1, n + 1))"]
    broken = ["return list(range(1, n))"]
    cases = [((1,), [1]), ((4,), [1, 2, 3, 4]), ((6,), [1, 2, 3, 4, 5, 6])]
    return fixed, broken, cases, ("range(1, n)", "range(1, n + 1)")


def _t_swapped_return(spec):
    fixed = ["return (a, b) if a <= b else (b, a)"]
    broken = ["return (b, a) if a <= b else (a, b)"]
    cases = [((2, 9), (2, 9)), ((9, 2), (2, 9)), ((5, 5), (5, 5)),
             ((-1, 3), (-1, 3))]
    return fixed, broken, cases, ("(b, a) if a <= b else (a, b)",
                                  "(a, b) if a <= b else (b, a)")


def _t_wrong_constant(spec):
    p = spec["k"]                            # discount percent
    wrong = p + 7 if p < 40 else p - 9
    fixed = [f"return price - price * {p} // 100"]
    broken = [f"return price - price * {wrong} // 100"]
    cases = [((1000,), 1000 - 1000 * p // 100), ((250,), 250 - 250 * p // 100),
             ((99,), 99 - 99 * p // 100), ((0,), 0)]
    return fixed, broken, cases, (f"price * {wrong}", f"price * {p}")


def _t_missing_edge_case(spec):
    fixed = ["if b == 0:", "    return 0", "return a // b"]
    broken = ["return a // b"]
    cases = [((10, 2), 5), ((7, 3), 2), ((9, 0), 0), ((0, 4), 0)]
    repair = ("return a // b", "if b == 0:\n        return 0\n    return a // b")
    return fixed, broken, cases, repair


def _t_inverted_comparison(spec):
    t = spec["k"]
    fixed = [f"return age >= {t}"]
    broken = [f"return age < {t}"]
    cases = [((t,), True), ((t + 5,), True), ((t - 1,), False)]
    return fixed, broken, cases, (f"age < {t}", f"age >= {t}")


def _t_wrong_variable(spec):
    fixed = ["return (a + b + c) // 3"]
    broken = ["return b"]
    cases = [((3, 6, 9), 6), ((0, 0, 12), 4), ((10, 20, 30), 20),
             ((5, 5, 5), 5)]
    return fixed, broken, cases, ("return b", "return (a + b + c) // 3")


def _t_missing_clamp(spec):
    fixed = ["if x < lo:", "    return lo", "if x > hi:", "    return hi",
             "return x"]
    broken = ["return x"]
    cases = [((5, 0, 10), 5), ((-3, 0, 10), 0), ((42, 0, 10), 10),
             ((0, 0, 10), 0)]
    repair = ("return x", "if x < lo:\n        return lo\n    if x > hi:"
                          "\n        return hi\n    return x")
    return fixed, broken, cases, repair


def _t_wrong_accumulator(spec):
    k = spec["k"]
    fixed = ["total = 0", "for v in values:", f"    total += v * {k}",
             "return total"]
    broken = ["total = 0", "for v in values:", f"    total = v * {k}",
              "return total"]
    cases = [(([1, 2, 3],), (1 + 2 + 3) * k), (([5],), 5 * k), (([],), 0),
             (([2, 2],), 4 * k)]
    return fixed, broken, cases, (f"total = v * {k}", f"total += v * {k}")


def _t_dropped_sign(spec):
    fixed = ["return abs(a - b)"]
    broken = ["return a - b"]
    cases = [((9, 4), 5), ((4, 9), 5), ((3, 3), 0), ((0, 7), 7)]
    return fixed, broken, cases, ("return a - b", "return abs(a - b)")


# (defect_type, builder, params)
_SLOTS = [
    ("wrong_operator", _t_wrong_operator, "x, y"),
    ("off_by_one", _t_off_by_one, "n"),
    ("swapped_return", _t_swapped_return, "a, b"),
    ("wrong_constant", _t_wrong_constant, "price"),
    ("missing_edge_case", _t_missing_edge_case, "a, b"),
    ("inverted_comparison", _t_inverted_comparison, "age"),
    ("wrong_variable", _t_wrong_variable, "a, b, c"),
    ("missing_clamp", _t_missing_clamp, "x, lo, hi"),
    ("wrong_accumulator", _t_wrong_accumulator, "values"),
    ("dropped_sign", _t_dropped_sign, "a, b"),
]

_NAME_POOL = [
    ("pricing", "calc_price"), ("scoring", "final_score"),
    ("billing", "invoice_total"), ("ranking", "rank_value"),
    ("metrics", "rollup"), ("ledger", "balance_after"),
    ("quotas", "allowance"), ("payroll", "net_pay"),
    ("shipping", "ship_cost"), ("loyalty", "points_for"),
]

_SYMPTOMS = {
    "wrong_operator": "results have the wrong sign or magnitude",
    "off_by_one": "the returned sequence is missing its last element",
    "swapped_return": "the returned pair comes back in the wrong order",
    "wrong_constant": "amounts are off by a consistent percentage",
    "missing_edge_case": "it crashes or misbehaves on a boundary input",
    "inverted_comparison": "the boolean comes back backwards",
    "wrong_variable": "the result ignores most of its inputs",
    "missing_clamp": "values outside the allowed range pass through unchanged",
    "wrong_accumulator": "only the last element seems to count",
    "dropped_sign": "results can be negative when they should not be",
}


def _check_src(module_file: str, func: str, cases) -> str:
    mod = module_file[:-3] if module_file.endswith(".py") else module_file
    lines = [f"from {mod} import {func}", ""]
    for args, expected in cases:
        call = f"{func}({', '.join(repr(a) for a in args)})"
        lines.append(f"assert {call} == {expected!r}, {call!r}")
    lines.append("print('ok')")
    return "\n".join(lines) + "\n"


def make_real_task(seed: int = 0, n_modules: int = 8,
                   difficulty: str = "standard") -> RealTask:
    """Generate a fresh, seeded real task instance (in-memory spec only).

    Use materialize() to lay it down in a sandbox directory.

    difficulty="standard" (the default) is the original generator, kept
    byte-for-byte reproducible. difficulty="hard" swaps in subtler,
    interacting and cross-module defects (see the hard section below);
    the module/check structure and result shapes are unchanged.
    """
    if difficulty == "hard":
        return _make_hard_task(seed, n_modules)
    if difficulty != "standard":
        raise ValueError(f"unknown difficulty: {difficulty!r} "
                         "(use 'standard' or 'hard')")
    rng = random.Random(seed ^ 0x5EED)
    slots = list(_SLOTS)
    rng.shuffle(slots)
    names = list(_NAME_POOL)
    rng.shuffle(names)
    modules = []
    for i in range(n_modules):
        defect_type, builder, params = slots[i % len(slots)]
        mod_name, func = names[i % len(names)]
        module_file = f"{mod_name}.py"
        spec = {"func": func,
                "k": rng.choice([3, 4, 5, 6, 8, 12, 15, 18, 21, 25])}
        fixed_lines, broken_lines, cases, repair = builder(spec)
        header = f"def {func}({params}):\n"
        fixed_code = header + "\n".join("    " + line for line in fixed_lines) + "\n"
        broken_code = header + "\n".join("    " + line for line in broken_lines) + "\n"
        find, repl = repair
        if not find or broken_code.replace(find, repl, 1) != fixed_code:
            raise AssertionError(f"repair pair mismatch for {defect_type}")
        modules.append(RealModule(
            subtask_id=f"rr-{i:02d}", module_file=module_file,
            check_file=f"check_{module_file}", func_name=func,
            defect_type=defect_type,
            description=(f"{module_file}: {func}() is failing its check -- "
                         f"{_SYMPTOMS[defect_type]}."),
            broken_code=broken_code, fixed_code=fixed_code,
            check_code=_check_src(module_file, func, cases), repair=repair))
    return RealTask(seed=seed, modules=modules)


# ------------------------------------------------------------ hard mode
# A second defect catalogue, selectable via difficulty="hard". The design
# goal: first-attempt success well below 100% for a strong small model,
# with difficulty coming from defect subtlety and interaction -- never
# from hidden information. The prompt, code and check scripts an agent
# sees are exactly as informative as in standard mode.
#
# Each builder takes (mod_name, func) for its main module and returns a
# dict with:
#   defect_type, symptom, n_defects
#   main:       (broken_code, fixed_code) for the main module file
#   extras:     {filename: (broken_code, fixed_code)} helper files
#   repair_file: which file the repair pair applies to ("" = main module)
#   repair:     (find, repl); target_broken.replace(find, repl, 1) must
#               equal target_fixed -- asserted at generation time
#   check_imports: import lines for the check script
#   probes:     check body blocks (no internal blank lines); each probe
#               is one independently runnable check unit
#   partial_fixes: [(filename, code)] fixing some but not all defects
#
# Composition is guaranteed per task: exactly one multi-defect module and
# at least one cross-module module, the rest drawn from the subtle pool.

def _h_boundary_search(mod_name, func):
    broken = (f"def {func}(values, cap):\n"
              "    best = -1\n"
              "    for i, v in enumerate(values):\n"
              "        if v < cap:\n"
              "            best = i\n"
              "    return best\n")
    fixed = broken.replace("if v < cap:", "if v <= cap:")
    return {
        "defect_type": "hard_boundary_search", "n_defects": 1,
        "symptom": "the returned index is wrong when a value exactly "
                   "equals the cap",
        "main": (broken, fixed), "extras": {}, "repair_file": "",
        "repair": ("if v < cap:", "if v <= cap:"),
        "check_imports": [f"from {mod_name} import {func}"],
        "probes": [
            f"assert {func}([3, 7, 2, 9], 7) == 2",
            f"assert {func}([5, 5, 5], 5) == 2",
            f"assert {func}([1, 2, 3], 0) == -1",
            f"assert {func}([8, 4, 8], 8) == 2",
            f"assert {func}([], 3) == -1",
            f"assert {func}([6, 1, 6, 9], 6) == 2",
        ],
        "partial_fixes": [],
    }


def _h_int_division(mod_name, func):
    broken = f"def {func}(a, b, c):\n    return (a + b + c) // 3\n"
    fixed = f"def {func}(a, b, c):\n    return (a + b + c) / 3\n"
    return {
        "defect_type": "hard_int_division", "n_defects": 1,
        "symptom": "averages come out truncated to whole numbers",
        "main": (broken, fixed), "extras": {}, "repair_file": "",
        "repair": ("(a + b + c) // 3", "(a + b + c) / 3"),
        "check_imports": [f"from {mod_name} import {func}"],
        "probes": [
            f"assert {func}(3, 6, 9) == 6",
            f"assert {func}(1, 2, 4) == 7 / 3",
            f"assert {func}(0, 0, 0) == 0",
            f"assert {func}(10, 20, 31) == 61 / 3",
            f"assert {func}(-1, 0, 1) == 0",
        ],
        "partial_fixes": [],
    }


def _h_sort_stability(mod_name, func):
    broken = (f"def {func}(entries):\n"
              "    return sorted(entries, key=lambda e: (-e[1], e[0]))\n")
    fixed = (f"def {func}(entries):\n"
             "    return sorted(entries, key=lambda e: -e[1])\n")
    return {
        "defect_type": "hard_sort_stability", "n_defects": 1,
        "symptom": "tied entries come back reordered instead of keeping "
                   "their original order",
        "main": (broken, fixed), "extras": {}, "repair_file": "",
        "repair": ("key=lambda e: (-e[1], e[0])", "key=lambda e: -e[1]"),
        "check_imports": [f"from {mod_name} import {func}"],
        "probes": [
            f"assert {func}([(\"bravo\", 9), (\"alpha\", 9), (\"delta\", 3)])"
            " == [(\"bravo\", 9), (\"alpha\", 9), (\"delta\", 3)]",
            f"assert {func}([(\"a\", 1), (\"b\", 2)]) == [(\"b\", 2), (\"a\", 1)]",
            f"assert {func}([(\"mike\", 5), (\"anna\", 5), (\"zoe\", 5)])"
            " == [(\"mike\", 5), (\"anna\", 5), (\"zoe\", 5)]",
            f"assert {func}([]) == []",
        ],
        "partial_fixes": [],
    }


def _h_mutable_default(mod_name, func):
    broken = (f"def {func}(item, bucket=[]):\n"
              "    bucket.append(item)\n"
              "    return bucket\n")
    fixed = (f"def {func}(item, bucket=None):\n"
             "    if bucket is None:\n"
             "        bucket = []\n"
             "    bucket.append(item)\n"
             "    return bucket\n")
    repair = (broken.rstrip("\n"), fixed.rstrip("\n"))
    return {
        "defect_type": "hard_mutable_default", "n_defects": 1,
        "symptom": "results leak items from earlier calls into later ones",
        "main": (broken, fixed), "extras": {}, "repair_file": "",
        "repair": repair,
        "check_imports": [f"from {mod_name} import {func}"],
        "probes": [
            f"r1 = {func}(\"a\")\nassert r1 == [\"a\"]\n"
            f"r2 = {func}(\"b\")\nassert r2 == [\"b\"]",
            f"r = {func}(\"x\", [\"seed\"])\nassert r == [\"seed\", \"x\"]",
            f"{func}(\"p\")\n{func}(\"q\")\nassert {func}(\"z\") == [\"z\"]",
        ],
        "partial_fixes": [],
    }


def _h_stale_cache(mod_name, func):
    broken = ("_cache = {}\n"
              "\n"
              f"def {func}(price, pct):\n"
              "    if price in _cache:\n"
              "        return _cache[price]\n"
              "    result = price - price * pct // 100\n"
              "    _cache[price] = result\n"
              "    return result\n")
    fixed = ("_cache = {}\n"
             "\n"
             f"def {func}(price, pct):\n"
             "    if (price, pct) in _cache:\n"
             "        return _cache[(price, pct)]\n"
             "    result = price - price * pct // 100\n"
             "    _cache[(price, pct)] = result\n"
             "    return result\n")
    find = ("    if price in _cache:\n"
            "        return _cache[price]\n"
            "    result = price - price * pct // 100\n"
            "    _cache[price] = result")
    repl = ("    if (price, pct) in _cache:\n"
            "        return _cache[(price, pct)]\n"
            "    result = price - price * pct // 100\n"
            "    _cache[(price, pct)] = result")
    return {
        "defect_type": "hard_stale_cache", "n_defects": 1,
        "symptom": "the same price with a different discount returns the "
                   "first discount's result",
        "main": (broken, fixed), "extras": {}, "repair_file": "",
        "repair": (find, repl),
        "check_imports": [f"from {mod_name} import {func}"],
        "probes": [
            f"assert {func}(200, 10) == 180\nassert {func}(200, 25) == 150",
            f"assert {func}(100, 10) == 90\nassert {func}(300, 10) == 270",
            f"assert {func}(150, 20) == 120\nassert {func}(150, 20) == 120",
        ],
        "partial_fixes": [],
    }


def _h_empty_input(mod_name, func):
    broken = (f"def {func}(values):\n"
              "    total = sum(values)\n"
              "    mean = total / len(values)\n"
              "    return {\"count\": len(values), \"total\": total, "
              "\"mean\": mean}\n")
    fixed = (f"def {func}(values):\n"
             "    total = sum(values)\n"
             "    mean = total / len(values) if values else 0.0\n"
             "    return {\"count\": len(values), \"total\": total, "
             "\"mean\": mean}\n")
    return {
        "defect_type": "hard_empty_input", "n_defects": 1,
        "symptom": "it blows up on an empty input instead of reporting "
                   "zeros",
        "main": (broken, fixed), "extras": {}, "repair_file": "",
        "repair": ("mean = total / len(values)",
                   "mean = total / len(values) if values else 0.0"),
        "check_imports": [f"from {mod_name} import {func}"],
        "probes": [
            f"assert {func}([2, 4]) == "
            "{\"count\": 2, \"total\": 6, \"mean\": 3.0}",
            f"assert {func}([]) == "
            "{\"count\": 0, \"total\": 0, \"mean\": 0.0}",
            f"assert {func}([5]) == "
            "{\"count\": 1, \"total\": 5, \"mean\": 5.0}",
            f"assert {func}([1, 2]) == "
            "{\"count\": 2, \"total\": 3, \"mean\": 1.5}",
        ],
        "partial_fixes": [],
    }


def _h_window_max(mod_name, func):
    broken = (f"def {func}(values, w):\n"
              "    return [max(values[i:i + w]) "
              "for i in range(len(values) - w)]\n")
    fixed = (f"def {func}(values, w):\n"
             "    return [max(values[i:i + w]) "
             "for i in range(len(values) - w + 1)]\n")
    return {
        "defect_type": "hard_window_off_by_one", "n_defects": 1,
        "symptom": "the last window is missing from the results",
        "main": (broken, fixed), "extras": {}, "repair_file": "",
        "repair": ("range(len(values) - w)", "range(len(values) - w + 1)"),
        "check_imports": [f"from {mod_name} import {func}"],
        "probes": [
            f"assert {func}([1, 5, 2], 3) == [5]",
            f"assert {func}([3, 1, 4, 1, 5], 2) == [3, 4, 4, 5]",
            f"assert {func}([2, 2], 1) == [2, 2]",
            f"assert {func}([7], 1) == [7]",
            f"assert {func}([], 1) == []",
        ],
        "partial_fixes": [],
    }


def _h_band_filter(mod_name, func):
    broken = (f"def {func}(values, lo, hi):\n"
              "    return [v for v in values if lo <= v <= hi]\n")
    fixed = (f"def {func}(values, lo, hi):\n"
             "    return [v for v in values if lo < v < hi]\n")
    return {
        "defect_type": "hard_band_boundaries", "n_defects": 1,
        "symptom": "values sitting exactly on the band edges are kept "
                   "when they should be dropped",
        "main": (broken, fixed), "extras": {}, "repair_file": "",
        "repair": ("lo <= v <= hi", "lo < v < hi"),
        "check_imports": [f"from {mod_name} import {func}"],
        "probes": [
            f"assert {func}([1, 2, 3, 4], 1, 4) == [2, 3]",
            f"assert {func}([2, 3], 1, 4) == [2, 3]",
            f"assert {func}([0, 5], 1, 4) == []",
            f"assert {func}([4, 1], 1, 4) == []",
            f"assert {func}([], 1, 4) == []",
        ],
        "partial_fixes": [],
    }


def _h_moving_average_multi(mod_name, func):
    broken = (f"def {func}(values, w):\n"
              "    return [sum(values[i:i + w]) // w "
              "for i in range(len(values) - w)]\n")
    fixed = (f"def {func}(values, w):\n"
             "    return [sum(values[i:i + w]) / w "
             "for i in range(len(values) - w + 1)]\n")
    only_div = (f"def {func}(values, w):\n"
                "    return [sum(values[i:i + w]) / w "
                "for i in range(len(values) - w)]\n")
    only_range = (f"def {func}(values, w):\n"
                  "    return [sum(values[i:i + w]) // w "
                  "for i in range(len(values) - w + 1)]\n")
    module_file = f"{mod_name}.py"
    return {
        "defect_type": "hard_multi_moving_average", "n_defects": 2,
        "symptom": "window averages are missing the last window and come "
                   "out truncated to whole numbers",
        "main": (broken, fixed), "extras": {}, "repair_file": "",
        "repair": ("[sum(values[i:i + w]) // w "
                   "for i in range(len(values) - w)]",
                   "[sum(values[i:i + w]) / w "
                   "for i in range(len(values) - w + 1)]"),
        "check_imports": [f"from {mod_name} import {func}"],
        "probes": [
            f"assert {func}([2, 4, 6], 3) == [4.0]",
            f"assert {func}([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]",
            f"assert {func}([3, 3, 3, 3], 2) == [3.0, 3.0, 3.0]",
            f"assert {func}([10, 0], 2) == [5.0]",
        ],
        "partial_fixes": [(module_file, only_div),
                          (module_file, only_range)],
    }


def _h_cross_units(mod_name, func):
    callee_broken = ("def to_cents(dollars):\n"
                     "    \"\"\"Convert a dollar amount to an integer "
                     "number of cents.\"\"\"\n"
                     "    return round(dollars * 10)\n")
    callee_fixed = callee_broken.replace("dollars * 10", "dollars * 100")
    caller = ("from money import to_cents\n"
              "\n"
              f"def {func}(items):\n"
              "    total = 0\n"
              "    for _name, price, qty in items:\n"
              "        total += to_cents(price) * qty\n"
              "    return total\n")
    return {
        "defect_type": "hard_cross_units", "n_defects": 1,
        "symptom": "totals come back far too small, as if the amounts "
                   "were in the wrong unit",
        "main": (caller, caller),
        "extras": {"money.py": (callee_broken, callee_fixed)},
        "repair_file": "money.py",
        "repair": ("return round(dollars * 10)",
                   "return round(dollars * 100)"),
        "check_imports": [f"from {mod_name} import {func}",
                          "from money import to_cents"],
        "probes": [
            f"assert {func}([(\"pen\", 1.5, 2)]) == 300",
            f"assert {func}([(\"book\", 10.0, 1), (\"pen\", 0.5, 4)])"
            " == 1200",
            f"assert {func}([]) == 0",
            "assert to_cents(2.5) == 250",
        ],
        "partial_fixes": [],
    }


def _h_cross_ordering(mod_name, func):
    callee_broken = ("def by_x(points):\n"
                     "    \"\"\"Return the points sorted by their x "
                     "coordinate, ascending.\"\"\"\n"
                     "    return sorted(points, key=lambda p: p[1])\n")
    callee_fixed = callee_broken.replace("key=lambda p: p[1]",
                                         "key=lambda p: p[0]")
    caller = ("from series import by_x\n"
              "\n"
              f"def {func}(points):\n"
              "    ordered = by_x(points)\n"
              "    total = 0.0\n"
              "    for (x1, y1), (x2, y2) in zip(ordered, ordered[1:]):\n"
              "        total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5\n"
              "    return total\n")
    return {
        "defect_type": "hard_cross_ordering", "n_defects": 1,
        "symptom": "path lengths are wrong for unordered point sets, as "
                   "if the points were visited in the wrong order",
        "main": (caller, caller),
        "extras": {"series.py": (callee_broken, callee_fixed)},
        "repair_file": "series.py",
        "repair": ("key=lambda p: p[1]", "key=lambda p: p[0]"),
        "check_imports": [f"from {mod_name} import {func}"],
        "probes": [
            f"assert abs({func}([(0, 2), (1, 0), (2, 2)])"
            " - 2 * 5 ** 0.5) < 1e-9",
            f"assert abs({func}([(0, 0), (1, 1), (2, 2)])"
            " - 2 * 2 ** 0.5) < 1e-9",
            f"assert {func}([(3, 4)]) == 0.0",
            f"assert abs({func}([(0, 1), (3, 0), (1, 2)])"
            " - 3 * 2 ** 0.5) < 1e-9",
        ],
        "partial_fixes": [],
    }


def _h_cross_shape(mod_name, func):
    callee_broken = ("def tally(values):\n"
                     "    \"\"\"Return (total, count) for the values.\"\"\"\n"
                     "    return (len(values), sum(values))\n")
    callee_fixed = callee_broken.replace(
        "return (len(values), sum(values))",
        "return (sum(values), len(values))")
    caller = ("from tally import tally\n"
              "\n"
              f"def {func}(values):\n"
              "    total, count = tally(values)\n"
              "    if count == 0:\n"
              "        return 0.0\n"
              "    return total / count\n")
    return {
        "defect_type": "hard_cross_shape", "n_defects": 1,
        "symptom": "averages are wildly off, as if the total and the "
                   "count were mixed up",
        "main": (caller, caller),
        "extras": {"tally.py": (callee_broken, callee_fixed)},
        "repair_file": "tally.py",
        "repair": ("return (len(values), sum(values))",
                   "return (sum(values), len(values))"),
        "check_imports": [f"from {mod_name} import {func}"],
        "probes": [
            f"assert {func}([2, 4, 6]) == 4.0",
            f"assert {func}([5]) == 5.0",
            f"assert {func}([]) == 0.0",
            f"assert {func}([10, 20]) == 15.0",
        ],
        "partial_fixes": [],
    }


_HARD_SINGLE = [_h_boundary_search, _h_int_division, _h_sort_stability,
                _h_mutable_default, _h_stale_cache, _h_empty_input,
                _h_window_max, _h_band_filter]
_HARD_CROSS = [_h_cross_units, _h_cross_ordering, _h_cross_shape]
_HARD_MULTI = [_h_moving_average_multi]


def _hard_check_src(imports, probes) -> str:
    lines = list(imports) + [""]
    for probe in probes:
        lines.append(probe)
        lines.append("")
    lines.append("print('ok')")
    return "\n".join(lines) + "\n"


def _make_hard_task(seed: int, n_modules: int) -> RealTask:
    """Seeded hard-mode task: subtle, multi-defect and cross-module
    subtasks in the same module/check shape as standard mode."""
    rng = random.Random(seed ^ 0x5EED ^ 0x4A2D)
    names = list(_NAME_POOL)
    rng.shuffle(names)
    # guaranteed composition: one multi-defect, at least one cross-module
    chosen = [rng.choice(_HARD_MULTI), rng.choice(_HARD_CROSS)]
    rest = list(_HARD_SINGLE) + [b for b in _HARD_CROSS if b not in chosen]
    rng.shuffle(rest)
    chosen += rest
    if n_modules <= len(chosen):
        builders = chosen[:n_modules]  # keeps the guaranteed composition
    else:
        builders = [chosen[i % len(chosen)] for i in range(n_modules)]
    rng.shuffle(builders)
    modules = []
    for i, builder in enumerate(builders):
        mod_name, func = names[i % len(names)]
        spec = builder(mod_name, func)
        module_file = f"{mod_name}.py"
        broken_code, fixed_code = spec["main"]
        # verify the scripted repair against the file it targets
        target = spec["repair_file"] or module_file
        pool = dict(spec["extras"])
        pool[module_file] = (broken_code, fixed_code)
        t_broken, t_fixed = pool[target]
        find, repl = spec["repair"]
        if not find or t_broken.replace(find, repl, 1) != t_fixed:
            raise AssertionError(
                f"repair pair mismatch for {spec['defect_type']}")
        description = (f"{module_file}: {func}() is failing its check -- "
                       f"{spec['symptom']}.")
        if spec["extras"]:
            description += (" The fault may sit in a helper module it "
                            "imports.")
        modules.append(RealModule(
            subtask_id=f"rr-{i:02d}", module_file=module_file,
            check_file=f"check_{module_file}", func_name=func,
            defect_type=spec["defect_type"], description=description,
            broken_code=broken_code, fixed_code=fixed_code,
            check_code=_hard_check_src(spec["check_imports"],
                                       spec["probes"]),
            repair=spec["repair"], extra_files=dict(spec["extras"]),
            repair_file=spec["repair_file"],
            n_defects=spec["n_defects"],
            partial_fixes=list(spec["partial_fixes"])))
    return RealTask(seed=seed, modules=modules, difficulty="hard")


def materialize(task: RealTask, sandbox: Path) -> Path:
    """Write the broken project (modules + check scripts) into `sandbox`."""
    sandbox = Path(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)
    for m in task.modules:
        write_in_sandbox(sandbox, m.module_file, m.broken_code)
        write_in_sandbox(sandbox, m.check_file, m.check_code)
        for fname, (extra_broken, _extra_fixed) in m.extra_files.items():
            write_in_sandbox(sandbox, fname, extra_broken)
    return sandbox


def run_check(sandbox: Path, module: RealModule, timeout: int = CHECK_TIMEOUT):
    """Ground truth for one subtask: (passed, output tail). Really executes.

    The sandbox __pycache__ is cleared first and the interpreter runs with
    -B: a fix written in the same second with the same byte length as the
    broken file would otherwise be shadowed by stale bytecode, and the
    check would grade the old file, not the agent's edit.
    """
    import shutil
    shutil.rmtree(Path(sandbox) / "__pycache__", ignore_errors=True)
    check = ensure_within(Path(sandbox), Path(module.check_file))
    try:
        proc = subprocess.run([sys.executable, "-B", str(check)], cwd=str(sandbox),
                              capture_output=True, text=True, timeout=timeout)
        out = (proc.stdout + proc.stderr).strip()
        return proc.returncode == 0, out[-1500:]
    except subprocess.TimeoutExpired:
        return False, f"check timed out after {timeout}s"


def check_all(task: RealTask, sandbox: Path) -> dict:
    """subtask_id -> passed, for every module, freshly executed."""
    return {m.subtask_id: run_check(sandbox, m)[0] for m in task.modules}
