from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import os, sys, re, random

# Data structures
@dataclass
class Node:
    name: str
    type: str
    fanin: List[str] = field(default_factory=list)
    fanout: List[str] = field(default_factory=list)
    is_input: bool = False
    is_output: bool = False

@dataclass
class Circuit:
    nodes: Dict[str, Node]
    inputs: List[str]
    outputs: List[str]
    topo: List[str]

# the following handles parsing of the circuit txt files
_GATE_EQ_RE = re.compile(r"(\w+)\s*=\s*(\w+)\(([^)]*)\)")

def _ensure(nodes: Dict[str, Node], name: str) -> Node:
    if name not in nodes:
        nodes[name] = Node(name=name, type="BUF", fanin=[])
    return nodes[name]

def parse_netlist(path: str) -> Circuit:
    nodes: Dict[str, Node] = {}
    inputs: List[str] = []
    outputs: List[str] = []

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            up = line.upper()

            # Input
            if up.startswith("INPUT"):
                if "(" in line:
                    name = line[line.find("(")+1:line.find(")")]
                    n = _ensure(nodes, name); n.is_input = True; n.type = "BUF"
                    inputs.append(name)
                else:
                    toks = line.split()
                    for t in toks[1:]:
                        if t == "-1":   # End of line reached
                            break
                        n = _ensure(nodes, t); n.is_input = True; n.type = "BUF"
                        inputs.append(t)
                continue

            # Output
            if up.startswith("OUTPUT"):
                if "(" in line:
                    name = line[line.find("(")+1:line.find(")")]
                    n = _ensure(nodes, name); n.is_output = True
                    outputs.append(name)
                else:
                    toks = line.split()
                    for t in toks[1:]:
                        if t == "-1":   # End of line reached
                            break
                        n = _ensure(nodes, t); n.is_output = True
                        outputs.append(t)
                continue

            # AND 9 5   or  NAND 1 2 7
            toks = line.split()
            if toks and toks[0].isalpha():
                typ = toks[0].upper()
                if len(toks) < 3:
                    raise ValueError(f"Bad gate line: {line}")
                out = toks[-1]
                fins = toks[1:-1]
                n = _ensure(nodes, out); n.type = typ; n.fanin = fins
                for a in fins:
                    _ensure(nodes, a).fanout.append(out)
                continue

            raise ValueError(f"Unrecognized line: {line}")

    # topo sort
    indeg = {name: 0 for name in nodes}
    for n in nodes.values():
        for fi in n.fanin:
            indeg[n.name] += 1
    q = [n for n in indeg if indeg[n] == 0]
    topo: List[str] = []
    while q:
        u = q.pop(0)
        topo.append(u)
        for v in nodes[u].fanout:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    return Circuit(nodes, inputs, outputs, topo)

# Logic evaluation functions
def gate_eval(gtype: str, ins: List[int]) -> int:
    g = gtype.upper()
    if g == "AND":
        return int(all(ins))
    if g == "NAND":
        return int(not all(ins))
    if g == "OR":
        return int(any(ins))
    if g == "NOR":
        return int(not any(ins))
    if g == "XOR":
        return sum(ins) % 2
    if g == "XNOR":
        return (sum(ins) + 1) % 2
    if g in ("INV", "NOT"):
        return 0 if ins[0] == 1 else 1
    if g in ("BUF", "WIRE"):
        return ins[0]
    raise ValueError(f"Unknown gate {gtype}")

def controlling_value(gtype: str) -> Optional[int]: #Defining controlling values for gates
    g = gtype.upper()
    if g in ("AND", "NAND"):
        return 0
    if g in ("OR", "NOR"):
        return 1
    return None

Fault = Tuple[str, int] #Data structure to store faults at nets

class DeductiveFaultSimulator:
    def __init__(self, circ: Circuit):
        self.c = circ
        self.net_list = list(circ.topo)
        self.U: Set[Fault] = {(n,0) for n in self.net_list} | {(n,1) for n in self.net_list}

    def simulate_good(self, pi_vals: Dict[str,int]) -> Dict[str,int]:
        v: Dict[str,int] = {}
        for pi in self.c.inputs:
            v[pi] = int(pi_vals[pi])
        for name in self.c.topo:
            node = self.c.nodes[name]
            if node.is_input:
                continue
            ins = [v[x] for x in node.fanin] if node.fanin else [0]
            v[name] = gate_eval(node.type, ins)
        return v

    def _xor_symdiff(self, lists: List[Set[Fault]]) -> Set[Fault]:
        counts: Dict[Fault, int] = {}
        for L in lists:
            for f in L:
                counts[f] = counts.get(f,0) + 1
        return {f for f,c in counts.items() if c % 2 == 1}

    def propagate_lists(self, good: Dict[str,int]) -> Dict[str,Set[Fault]]:
        L = {name:set() for name in self.c.nodes}
        # PIs
        for pi in self.c.inputs:
            L[pi].add((pi, 1 - good[pi]))
        # internal
        for name in self.c.topo:
            node = self.c.nodes[name]
            if node.is_input:
                continue
            gv = good[name]
            g = node.type.upper()
            if g in ("BUF","WIRE","INV","NOT"):
                L[name] = set(L[node.fanin[0]])
            elif g in ("AND","NAND","OR","NOR"):
                c = controlling_value(g)
                C = [x for x in node.fanin if good[x] == c]
                if len(C) == 0:
                    U = set()
                    for x in node.fanin:
                        U |= L[x]
                    L[name] = U
                else:
                    inter = None
                    for x in C:
                        inter = L[x] if inter is None else inter & L[x]
                    inter = inter or set()
                    others = set()
                    for x in node.fanin:
                        if x not in C:
                            others |= L[x]
                    L[name] = inter - others
            elif g in ("XOR","XNOR"):
                L[name] = self._xor_symdiff([L[x] for x in node.fanin])
            else:
                acc = set()
                for x in node.fanin:
                    acc |= L[x]
                L[name] = acc
            # add output fault
            L[name].add((name, 1 - gv))
        return L

    def bits_to_map(self, bits: str) -> Dict[str,int]:
        if len(bits) != len(self.c.inputs):
            raise ValueError("vector length mismatch")
        return {pi:int(b) for pi,b in zip(self.c.inputs, bits)}

    def detect_for_vector(self, bits: str, restrict: Optional[Set[Fault]]=None) -> Set[Fault]:
        good = self.simulate_good(self.bits_to_map(bits))
        L = self.propagate_lists(good)
        det = set()
        for po in self.c.outputs:
            det |= L[po]
        if restrict is not None:
            det &= restrict
        return det

    def coverage(self, detected: Set[Fault], universe: Optional[Set[Fault]]=None) -> float:
        U = universe if universe is not None else self.U
        return 100.0 * len(detected & U) / len(U)

    def write_report(self, out_path: str, vectors: List[str], detected: Set[Fault], universe: Optional[Set[Fault]]=None):
        U = universe if universe is not None else self.U
        with open(out_path, "w") as f:
            f.write("-------------------------------------------\n")
            f.write(f"Circuit name: {os.path.splitext(os.path.basename(out_path))[0].replace('_output','')}\n")


            # If vectors applied is one, then show the vector otherwise showing the number of vectors applied
            if len(vectors) == 1:
                f.write(f"Vector applied: {vectors[0]}\n\n")
            else:
                f.write(f"Vectors applied ({len(vectors)} total):\n")
                for v in vectors:
                    f.write(f"  {v}\n")
                f.write("\n")

            f.write("Detected faults (net stuck-at):\n")
            for net, sa in sorted(detected, key=lambda x: (int(x[0]), x[1])):
                f.write(f"{net} s-a-{sa}\n")

            f.write(f"\nDetected {len(detected & U)} / {len(U)}   Coverage={self.coverage(detected, U):.2f}%\n")
            f.write("-------------------------------------------\n")

# Parsing fault list from fault list file
def parse_fault_list_file(net_order: List[str], path: str) -> Set[Fault]:
    S: Set[Fault] = set()
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            num, sval = s.split()[:2]
            idx = int(num) - 1
            val = int(sval)
            if not (0 <= idx < len(net_order)):
                raise ValueError(f"Bad net no in fault file: {num}")
            S.add((net_order[idx], val))
    return S

# interactive_main() handles the itneraction between user and the program via the terminal
def interactive_main():
    print("ECE6140 Digital Systems Test: Deductive Fault Simulator")
    base = input("Enter circuit file name (without .txt): ").strip()
    netlist = base + ".txt"
    if not os.path.exists(netlist):
        print("File not found:", netlist); return

    circ = parse_netlist(netlist)
    sim = DeductiveFaultSimulator(circ)

    print("\n1) Simulate ALL faults (give vector file)")
    print("2) Simulate ONLY faults listed in a file")
    print("3) RANDOM mode (start 5, double until limit, make coverage data)")
    mode = input("Choose option (1/2/3): ").strip()

    if mode == "1":
        vbase = input("Enter vector file name (without .txt): ").strip()
        vfile = vbase + ".txt"
        if not os.path.exists(vfile):
            print("Vector file not found:", vfile); return
        vectors = [line.strip() for line in open(vfile) if line.strip()]
        detected: Set[Fault] = set()
        for v in vectors:
            detected |= sim.detect_for_vector(v)
        out = base + "_output.txt"
        sim.write_report(out, vectors, detected)
        print("Wrote", out)

    elif mode == "2":
        vbase = input("Enter vector file name (without .txt): ").strip()
        vfile = vbase + ".txt"
        fbase = input("Enter fault file name (without .txt): ").strip()
        ffile = fbase + ".txt"
        if not os.path.exists(vfile) or not os.path.exists(ffile):
            print("Vector or fault file not found."); return
        vectors = [line.strip() for line in open(vfile) if line.strip()]
        restrict = parse_fault_list_file(sim.net_list, ffile)
        detected: Set[Fault] = set()
        for v in vectors:
            detected |= sim.detect_for_vector(v, restrict)
        out = base + "_output.txt"
        sim.write_report(out, vectors, detected, universe=restrict)
        print("Wrote", out)

    elif mode == "3":
        try:
            end_val = int(input("Enter max number of random tests (e.g. 500): ").strip())
        except ValueError:
            print("Bad number."); return
        seed_s = input("Enter seed (default 1): ").strip()
        seed = int(seed_s) if seed_s else 1
        random.seed(seed)

        counts: List[int] = []
        covs: List[float] = []
        detected: Set[Fault] = set()
        vecs: List[str] = []

        for cur in range(1, end_val + 1):
            # add one new random vector each iteration
            vec = "".join(str(random.randint(0,1)) for _ in circ.inputs)
            vecs.append(vec)

            # simulate new vector only and update detected set
            detected |= sim.detect_for_vector(vec)

            cov = sim.coverage(detected)
            counts.append(cur)
            covs.append(cov)
            print(f"{cur} tests -> {cov:.2f}% coverage")


        # write CSV to plot later
        out_csv = base + "_coverage.txt"
        with open(out_csv, "w") as f:
            f.write("tests,coverage\n")
            for n, c in zip(counts, covs):
                f.write(f"{n},{c:.2f}\n")
        print("Coverage data written to", out_csv)

        # Saving plots
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(counts, covs, marker="o")
            plt.xlabel("No. of applied random tests")
            plt.ylabel("Fault coverage (%)")
            plt.title(f"Coverage vs tests: {base}.txt")
            plt.grid(True)
            plot_name = base + "_coverage.png"
            plt.savefig(plot_name)
            print("Plot saved to", plot_name)
        except Exception:
            print("matplotlib not available – use the CSV to plot in Excel.")

    else:
        print("Unknown option.")
def main():
    if len(sys.argv) == 1:
        interactive_main()
        return
    # run directly thourgh command prompt
    interactive_main()

if __name__ == "__main__":
    main()