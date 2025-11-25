"""
PODEM-based ATPG for project part 3.

This module is designed to plug in along with
`deductive_fault_sim.py`.

- It uses `deductive_fault_sim.parse_netlist` to read the ISCAS-style netlist.
- It implements a 5-valued logic PODEM search (0, 1, X, D, ~D) on top of that
  circuit representation.
- It can verify each generated test vector using the existing
  `DeductiveFaultSimulator` class.

Usage (example):

    python podem_atpg.py s27.txt 16/0 10/1 12/0 18/1

where each fault is given as "<net>/<stuck_value>":
    - net : name/ID of the net in the netlist file (e.g. "16")
    - stuck_value: 0 or 1 (for s-a-0 or s-a-1)

The program prints, for each fault:
    - the generated test vector in INPUT order (as a 0/1/X string), or
    - reports the fault as UNDETECTABLE
    - and whether the vector is verified by the deductive fault simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import sys

# Reusing my simulator code
import deductive_fault_sim as dfs


# 5-valued logic

# Represent 5-valued logic as simple strings:
#   '0', '1', 'X', 'D', 'B'  (where 'B' is ~D = 0/1)
V0 = '0'
V1 = '1'
VX = 'X'
VD = 'D'
VDBAR = 'B'   # ~D


def _v5_to_pair(v: str) -> Tuple[Optional[int], Optional[int]]:
    """Map 5-valued symbol -> (good, faulty) pair."""
    if v == V0:
        return 0, 0
    if v == V1:
        return 1, 1
    if v == VD:
        return 1, 0
    if v == VDBAR:
        return 0, 1
    # unknown
    return None, None


def _pair_to_v5(g: Optional[int], f: Optional[int]) -> str:
    """Map (good, faulty) pair -> 5-valued symbol."""
    if g is None or f is None:
        return VX
    if g == 0 and f == 0:
        return V0
    if g == 1 and f == 1:
        return V1
    if g == 1 and f == 0:
        return VD
    if g == 0 and f == 1:
        return VDBAR
    # should not happen
    return VX


def _eval_3val(gtype: str, ins: List[Optional[int]]) -> Optional[int]:
    """
    3-valued gate evaluation with values in {0,1,None} (None = X).
    This is used separately for the good and faulty circuits.
    """
    g = gtype.upper()

    def all_known(vals: List[Optional[int]]) -> bool:
        for v in vals:
            if v is None:
                return False
        return True

    if g in ("BUF", "WIRE"):
        return ins[0]

    if g in ("INV", "NOT"):
        v = ins[0]
        if v is None:
            return None
        return 0 if v == 1 else 1

    if g == "AND":
        if any(v == 0 for v in ins):
            return 0
        if all_known(ins) and all(v == 1 for v in ins):
            return 1
        return None

    if g == "NAND":
        # NAND = NOT(AND)
        base = _eval_3val("AND", ins)
        if base is None:
            return None
        return 0 if base == 1 else 1

    if g == "OR":
        if any(v == 1 for v in ins):
            return 1
        if all_known(ins) and all(v == 0 for v in ins):
            return 0
        return None

    if g == "NOR":
        base = _eval_3val("OR", ins)
        if base is None:
            return None
        return 0 if base == 1 else 1

    if g == "XOR":
        # XOR: parity of 1s; if any X, output is X
        if any(v is None for v in ins):
            return None
        return sum(1 for v in ins if v == 1) % 2

    if g == "XNOR":
        base = _eval_3val("XOR", ins)
        if base is None:
            return None
        return 0 if base == 1 else 1

    raise ValueError(f"Unknown gate type: {gtype}")


# PODEM ATPG core algorithm

Fault = dfs.Fault  # Tuple[str, int]


@dataclass
class PODEMEngine:
    circ: dfs.Circuit
    fault_node: str
    stuck_at: int  # 0 or 1

    def __post_init__(self) -> None:
        if self.stuck_at not in (0, 1):
            raise ValueError("stuck_at must be 0 or 1")
        if self.fault_node not in self.circ.nodes:
            raise ValueError(f"Unknown net '{self.fault_node}' in circuit")

        # Depth and call limits to avoid getting stuck on undetectable faults
        self.max_depth = 2 * len(self.circ.inputs) 
        self.max_calls = 50000                       # global recursion limit
        self._calls = 0

    # implication

    def imply(self, pi_assign: Dict[str, int]) -> Tuple[Dict[str, str], Set[str]]:

        # Forward 5-valued implication from PI assignments.
        # Returns:
        #     values    : node -> 5-valued symbol
        #     d_frontier: set of gate-output node names currently on D-frontier

        values: Dict[str, str] = {}

        # 1) primary inputs
        for name in self.circ.inputs:
            bit = pi_assign.get(name, None)
            if bit is None:
                g = f = None
            else:
                g = f = int(bit)
            if name == self.fault_node:
                # inject stuck-at fault at the fault site
                f = self.stuck_at
            values[name] = _pair_to_v5(g, f)

        # 2) all other nodes in topological order
        for name in self.circ.topo:
            node = self.circ.nodes[name]
            if node.is_input:
                # already assigned in step 1
                continue
            # fanin 3-valued pairs
            in_pairs = [_v5_to_pair(values[fi]) for fi in node.fanin]
            in_good = [g for (g, _) in in_pairs]
            in_fault = [f for (_, f) in in_pairs]

            gg = _eval_3val(node.type, in_good) if in_good else None
            gf = _eval_3val(node.type, in_fault) if in_fault else None

            if name == self.fault_node:
                gf = self.stuck_at

            values[name] = _pair_to_v5(gg, gf)

        # 3) compute D-frontier
        d_frontier: Set[str] = set()
        for name in self.circ.topo:
            node = self.circ.nodes[name]
            if node.is_input or not node.fanin:
                continue

            out_v = values[name]
            if out_v != VX:
                continue

            in_vals = [values[fi] for fi in node.fanin]
            d_like = [v for v in in_vals if v in (VD, VDBAR)]
            if not d_like:
                continue

            gtype = node.type.upper()
            if gtype == "XOR":
                # XOR sensitized iff odd number of D/~D inputs
                if len(d_like) % 2 == 1:
                    d_frontier.add(name)
            else:
                d_frontier.add(name)

        return values, d_frontier

    # PODEM support routines

    def _error_at_po(self, values: Dict[str, str]) -> bool:
        #Check if D / ~D appears at any primary output.
        return any(values[o] in (VD, VDBAR) for o in self.circ.outputs)

    def _all_pis_assigned(self, pi_assign: Dict[str, int]) -> bool:
        #Check if every PI currently has a 0/1 assignment.
        return all(pi in pi_assign for pi in self.circ.inputs)

    def _x_path_exists(self, start: str, values: Dict[str, str]) -> bool:

        stack = [start]
        visited: Set[str] = set()
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            node = self.circ.nodes[u]
            for v in node.fanout:
                if v in visited:
                    continue
                if values[v] != VX:
                    continue
                if self.circ.nodes[v].is_output:
                    return True
                stack.append(v)
        return False

    def _objective(
        self,
        values: Dict[str, str],
        d_frontier: Set[str],
    ) -> Optional[Tuple[str, int]]:

        # Compute the next objective (line, 0/1) or return None if no objective.

        # - If the fault site has not yet produced D/~D, try to activate the fault.
        # - Once activated, pick a gate from the D-frontier and try to
        #   propagate the error by setting a non-controlling value on one input.

        v_fault = values[self.fault_node]

        # 1) Fault activation objective
        if v_fault not in (VD, VDBAR):
            # want the GOOD value at the fault site to be 1-stuck_at
            desired = 1 - self.stuck_at
            return self.fault_node, desired

        # 2) fault already activated – drive D / ~D towards a PO
        if not d_frontier:
            return None

        # pick the first gate on D-frontier
        gname = next(iter(d_frontier))
        gnode = self.circ.nodes[gname]
        gtype = gnode.type.upper()

        # choose an input whose current value is X if possible
        x_inputs = [fi for fi in gnode.fanin if values[fi] == VX]
        target_in = x_inputs[0] if x_inputs else gnode.fanin[0]

        if gtype in ("XOR", "XNOR"):
            # For XOR/XNOR there is no strict controlling value.
            # As a simple rule, keep remaining inputs at 0 first.
            return target_in, 0

        c = dfs.controlling_value(gtype)
        if c is None:
            # BUF/INV: just push through; actual inversion is
            # handled in backtrace.
            return target_in, 1  # arbitrary; backtrace will flip as needed

        non_c = 1 - c
        return target_in, non_c

    def _backtrace(
        self,
        line: str,
        val: int,
        values: Dict[str, str],
    ) -> Tuple[str, int]:

        # Backtrace an objective (line,val) to some PI (pi,val_pi).

        # Move backwards in the fanin cone, flipping the desired value at each
        # inverting gate (INV, NAND, NOR, XNOR), preferring fanins that
        # are still X to keep the search space small.
  
        cur = line
        desired = val

        while cur not in self.circ.inputs:
            node = self.circ.nodes[cur]
            gtype = node.type.upper()

            # Determine inversion parity of this gate
            inverted = gtype in ("INV", "NOT", "NAND", "NOR", "XNOR")
            if inverted:
                desired ^= 1  # flip 0/1

            # Pick next line to trace through: prefer X-valued fanin
            x_inputs = [fi for fi in node.fanin if values[fi] == VX]
            if x_inputs:
                cur = x_inputs[0]
            elif node.fanin:
                cur = node.fanin[0]
            else:

                break

        return cur, desired

    # Main recursive PODEM search

    def podem(
        self,
        pi_assign: Optional[Dict[str, int]] = None,
        depth: int = 0,
    ) -> Tuple[bool, Optional[Dict[str, int]]]:

        # Recursive PODEM search.

        # pi_assign: current partial PI assignment: PI -> 0/1


        # global call-count cut-off
        self._calls += 1
        if self._calls > self.max_calls:
            return False, None

        # depth cut-off to avoid infinite recursion
        if depth > self.max_depth:
            return False, None

        if pi_assign is None:
            pi_assign = {}

        values, d_frontier = self.imply(pi_assign)

        # SUCCESS: error is visible at some primary output
        if self._error_at_po(values):
            return True, pi_assign

        # If all PIs are assigned and no error at PO, this branch fails
        if self._all_pis_assigned(pi_assign):
            return False, None

        # Compute next objective
        obj = self._objective(values, d_frontier)
        if obj is None:
            # no further propagation possible under this assignment
            return False, None

        line, target_val = obj

        # Backtrace objective to a PI
        pi, pi_val = self._backtrace(line, target_val, values)

        # Branch 1: pi = pi_val
        if pi in pi_assign and pi_assign[pi] == pi_val:
            ok, sol = self.podem(dict(pi_assign), depth + 1)
            if ok:
                return True, sol
        else:
            assign1 = dict(pi_assign)
            assign1[pi] = pi_val
            ok, sol = self.podem(assign1, depth + 1)
            if ok:
                return True, sol

        # Branch 2: pi = 1 - pi_val
        alt_val = 1 - pi_val
        if pi in pi_assign and pi_assign[pi] == alt_val:
            return False, None
        assign2 = dict(pi_assign)
        assign2[pi] = alt_val
        ok, sol = self.podem(assign2, depth + 1)
        if ok:
            return True, sol

        # No assignment worked at this level
        return False, None

    # PODEM starter function

    def find_test_vector(self) -> Optional[str]:

        # Run PODEM and, if successful, return a 0/1/X string of length PIs
        # in the circuit's INPUT order. If no test exists, return None.

        ok, sol = self.podem({})
        if not ok or sol is None:
            return None

        # Map PIs into a dense 0/1/X vector in input order; unassigned -> X
        bits = []
        for pi in self.circ.inputs:
            if pi in sol:
                bits.append(str(sol[pi]))
            else:
                bits.append('X')
        return "".join(bits)


# Terminal input acceptor

def _parse_fault_token(tok: str) -> Fault:

    # Parse a fault token of the form "net/0" or "net/1" and return (net, stuck).

    if "/" not in tok:
        raise ValueError(
            f"Fault must be given as net/stuck, e.g. '16/0' – got '{tok}'"
        )
    net, sval = tok.split("/", 1)
    net = net.strip()
    if sval not in ("0", "1"):
        raise ValueError(f"stuck value must be 0 or 1, got '{sval}'")
    return net, int(sval)


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        prog = argv[0] if argv else "podem_atpg.py"
        print(f"Usage: {prog} <netlist.txt> <fault1> [fault2 ...]", file=sys.stderr)
        print(
            "  where each fault is given as net/stuck, e.g. 16/0 for 'Net 16 s-a-0'",
            file=sys.stderr,
        )
        return 2

    netlist_path = argv[1]
    fault_tokens = argv[2:]

    try:
        circ = dfs.parse_netlist(netlist_path)
    except Exception as e:
        print(f"Error parsing netlist '{netlist_path}': {e}", file=sys.stderr)
        return 1

    # Deductive fault simulator for verification
    try:
        sim = dfs.DeductiveFaultSimulator(circ)
    except Exception:
        sim = None

    print(f"# Loaded circuit: {netlist_path}")
    print(f"# PIs: {circ.inputs}")
    print(f"# POs: {circ.outputs}")
    print()

    for tok in fault_tokens:
        try:
            net, stuck = _parse_fault_token(tok)
        except Exception as e:
            print(f"[{tok}] parse error: {e}")
            continue

        print(f"Fault {net} s-a-{stuck}:")

        try:
            engine = PODEMEngine(circ, net, stuck)
        except Exception as e:
            print(f"  error: {e}")
            print()
            continue

        # Run PODEM; treat recursion exhaustion as "undetectable"
        try:
            vec = engine.find_test_vector()
        except RecursionError:
            print("  UNDETECTABLE (PODEM recursion limit reached)")
            print()
            continue

        if vec is None:
            print("  UNDETECTABLE (no test vector found by PODEM)")
            print()
            continue

        print(f"  Test vector (PI order {circ.inputs}): {vec}")

        # vec may have X's; use a 0/1-only version for dfs
        vec_for_sim = vec.replace('X', '0')

        # Verification via deductive fault simulator
        if sim is not None:
            try:
                detected = sim.detect_for_vector(vec_for_sim, restrict={(net, stuck)})
                if (net, stuck) in detected:
                    print("  Verified by deductive fault simulator: DETECTS fault.")
                else:
                    print(
                        "  WARNING: deductive fault simulator did NOT report this "
                        "fault as detected."
                    )
            except Exception as e:
                print(f"  (verification failed: {e})")

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))