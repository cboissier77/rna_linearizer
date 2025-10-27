# rna_linear_designer.py
import random
import re
from typing import List, Tuple, Dict
import math

import streamlit as st
import RNA  # ViennaRNA python bindings


# --------------------------- Core utilities ---------------------------

def construct_rna_sequence(
    head: str,
    motif: str,
    tail: str,
    repetition_motif: int,
    len_intermotifs: int,
    padding_front: int = 0,
    padding_end: int = 0,
) -> str:
    """
    Construct an RNA sequence template with 'N' placeholders where designable.
    Pattern: head + (N * padding_front) + [motif + (N * len_intermotifs)]*(repetition_motif-1)
             + motif + (N * padding_end) + tail
    """
    return (
        head
        + ("N" * padding_front)
        + ((motif + ("N" * len_intermotifs)) * max(repetition_motif - 1, 0))
        + motif
        + ("N" * padding_end)
        + tail
    )


def pu_vector(seq): 
    """Return per-base unpaired probabilities from base-pair probabilities.""" 
    fc = RNA.fold_compound(seq) 
    fc.pf() 
    bpp = fc.bpp() 
    n = len(seq) 
    pu = [1.0]*n 
    for i in range(n): 
        for j in range(1, n+1): 
            if i+1 < j: 
                pu[i] -= bpp[i+1][j] 
            elif i+1 > j: 
                pu[i] -= bpp[j][i+1] 
    return pu

def hard_constraint_vector(seq: str, roi_start: int, roi_end: int) -> List[bool]:
    """
    Return True if the RNA sequence does NOT contain unwanted motifs,
    False otherwise.
    """
    seq_to_check = seq[roi_start : roi_end + 1]

    # Regex patterns (compiled for efficiency)
    patterns = {
        # start codon
        "start_codon": re.compile(r"AUG"),
        # stop codons
        "stop_codon": re.compile(r"UAA|UAG|UGA"),
        # Homopolymers ≥6
        "homopolymer": re.compile(r"A{4,}|U{4,}|G{4,}|C{4,}"),
        # PolyA-like signals
        "polyA_signal": re.compile(r"AAUAAA|AUUAAA|AAGAAA|AAUAUA|UAUAAA|AAAUA|AAGAA"),
        # Splice donor (loose)
        "splice_donor": re.compile(r"[AG]AGG[U][AG]AGU"),
        # Splice acceptor (loose)
        "splice_acceptor": re.compile(r"[UC]{8,12}[ACGU]?CAGG"),
        # AU-rich elements (AREs)
        "AREs": re.compile(r"(?:AUUUA){2,}"),  # repeated AUUUA motifs
        # G-quadruplex motif
        "G4": re.compile(r"G{3,}\w{1,7}G{3,}\w{1,7}G{3,}\w{1,7}G{3,}")
    }

    # Search each pattern
    for name, pattern in patterns.items():
        if pattern.search(seq_to_check):
            return False

    return True

def check_no_additional_motif(seq: str, motif: str, roi_start: int, roi_end: int, number_motifs: int) -> bool:
    """
    Check if the RNA sequence contains the specified number of non-overlapping instances of a motif
    in the region of interest (ROI).
    """
    # Extract the region of interest
    roi = seq[roi_start:roi_end + 1]
    # Count non-overlapping occurrences of the motif
    count = sum(1 for _ in re.finditer(rf"(?={motif})", roi))
    return count == number_motifs

def score_pu(seq: str, start: int, end: int) -> Tuple[float, List[float]]:
    """
    Compute sum of log PU in the region of interest (inclusive).
    """
    pu = pu_vector(seq)
    roi = pu[start : end + 1]
    sum_log = sum((0.0 if p == 0.0 else math.log(p) for p in roi))
    return sum_log, pu


def mutate_one(seq: str, pos: int, alphabet: Tuple[str, ...] = ("A", "U", "G", "C")) -> str:
    current = seq[pos]
    choices = [b for b in alphabet if b != current]
    return seq[:pos] + random.choice(choices) + seq[pos + 1:]


def optimize_slice(
    full_seq: str,
    design_mask: List[bool],
    motif: str,
    repetition_motif: int,
    roi_start: int,
    roi_end: int,
    constraint_roi_start: int,
    constraint_roi_end: int,
    iters: int,
    batch_size: int = 5
) -> Tuple[str, float]:
    """
    Greedy hill-climb: at each step mutate the designable position with the lowest PU.
    Operates on the *full* sequence but only mutates positions where design_mask[i] is True.

    Returns (best_seq, best_mean_pu_in_roi).
    """
    if batch_size > sum(design_mask):
        batch_size = sum(design_mask)
    seq = full_seq
    best_seq = seq
    best_score, _ = score_pu(seq, roi_start, roi_end)
    alphabet = ("A", "U", "G", "C")
    # every 50 muated all
    mutate_all = 400
    scores = []


    for i in range(max(0, iters)):
        score, pu = score_pu(seq, roi_start, roi_end)
        scores.append(score)
        if score > best_score:
            best_score, best_seq = score, seq
        if i % mutate_all == 0:
            # mutate all designable positions once
            for pos in range(len(seq)):
                if design_mask[pos]:
                    seq = mutate_one(seq, pos, alphabet=alphabet)
            while not hard_constraint_vector(seq, constraint_roi_start,  constraint_roi_end) or not check_no_additional_motif(seq, motif, constraint_roi_start, constraint_roi_end, repetition_motif):
                for pos in range(len(seq)):
                    if design_mask[pos]:
                        seq = mutate_one(seq, pos, alphabet=alphabet)           

        # pick batch size of designable position with lowest PU
        else:
            candidates = [(i, pu[i]) for i in range(len(seq)) if design_mask[i]]
            candidates.sort(key=lambda x: x[1])  # sort by PU
            candidates = candidates[:batch_size]
            n_to_mut = random.randint(1, batch_size)
            # select n_to_mut randomly from the batch
            candidates = random.sample(candidates, n_to_mut)
            # select one randomly from the batch
            #min_pos = random.choice(candidates)[0]
            for pos, _ in candidates:
                seq = mutate_one(seq, pos, alphabet=alphabet)

            while not hard_constraint_vector(seq, constraint_roi_start, constraint_roi_end) or not check_no_additional_motif(seq, motif, constraint_roi_start, constraint_roi_end, repetition_motif):
                min_pos = random.choice(candidates)[0]
                seq = mutate_one(seq, min_pos, alphabet=alphabet)

    return best_seq, best_score, scores

def initial_fill_from_template(template: str) -> str:
    """
    Replace 'N' with random nucleotides (uniform, adjust weights if you want AU bias).
    """
    alphabet = ("A", "U", "G", "C")
    return "".join(random.choice(alphabet) if c == "N" else c for c in template)


def staged_optimize_linear_rna(
    motif: str,
    head: str,
    tail: str,
    repetition_motif: int,
    len_intermotifs: int,
    padding_front: int = 0,
    padding_end: int = 0,
    iters: int = 5000,
) -> Dict[str, object]:
    """
    Iterative optimization.
    """
    # Build template and initial guess
    template = construct_rna_sequence(
        head, motif, tail, repetition_motif, len_intermotifs, padding_front, padding_end
    )

    seq = initial_fill_from_template(template)

    n = len(seq)
    Lh, Lt = len(head), len(tail)
    # ROI: the "linearize" region is the part between head and tail
    roi_start = Lh + padding_front
    roi_end = n - Lt - padding_end - 1
    constraint_roi_start = Lh
    constraint_roi_end = n - Lt - 1

    while not hard_constraint_vector(seq, constraint_roi_start, constraint_roi_end) or not check_no_additional_motif(seq, motif, constraint_roi_start, constraint_roi_end, repetition_motif):
        seq = initial_fill_from_template(template)

    design_mask = [c == "N" for c in template]

    # Optimize this phase on the full sequence (mutations constrained by mask)
    seq , _, scores = optimize_slice(
        full_seq=seq,
        design_mask=design_mask,
        motif=motif,
        repetition_motif=repetition_motif,
        roi_start=roi_start,
        roi_end=roi_end,
        constraint_roi_start=constraint_roi_start,
        constraint_roi_end=constraint_roi_end,
        iters=iters,
    )

    # Final evaluation
    fc = RNA.fold_compound(seq)
    mfe_struct, mfe_energy = fc.mfe()
    pu = pu_vector(seq)
    mean_pu_all = sum(pu) / len(pu)
    mean_pu_linear = sum(pu[roi_start : roi_end + 1]) / max(1, (roi_end - roi_start + 1))
    min_pu_linear = min(pu[roi_start : roi_end + 1]) if roi_end >= roi_start else 0.0

    return {
        "sequence": seq,
        "mfe_structure": mfe_struct,
        "mfe_kcal_per_mol": float(mfe_energy),
        "mean_pu_all": float(mean_pu_all),
        "linearize_seq": seq[constraint_roi_start : constraint_roi_end + 1],
        "mean_pu_linear": float(mean_pu_linear),
        "min_pu_linear": float(min_pu_linear),
        "roi_start": roi_start,
        "roi_end": roi_end,
        "template": template,
        "scores": scores,
    }


# --------------------------- Streamlit UI ---------------------------

st.set_page_config(page_title="RNA Linear Region Designer", layout="wide")
st.title("🧬 RNA Linear Region Designer")

st.markdown(
    "Design an RNA sequence with a **linear (unpaired)** region of interest (ROI) using the **ViennaRNA** model."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Input parameters")
    head = st.text_input("Head (fixed)", "AAA").upper().strip()
    motif = st.text_input("Motif", "AAAGGG").upper().strip()
    tail = st.text_input("Tail (fixed)", "UUU").upper().strip()
    repeats = st.number_input("Motif repeats", min_value=1, max_value=50, value=3, step=1)
    Ns_between = st.number_input("N between motifs", min_value=0, max_value=200, value=5, step=1)
    padding_front = st.number_input("Padding front (N)", min_value=0, max_value=1000, value=0, step=1)
    padding_end = st.number_input("Padding end (N)", min_value=0, max_value=1000, value=0, step=1)
    iters = st.number_input("Iterations", min_value=10, max_value=1_000_000, value=400, step=10)
    run_button = st.button("🚀 Design RNA", type="primary")

with col2:
    st.subheader("Constructed RNA template")
    template = construct_rna_sequence(head, motif, tail, repeats, Ns_between, padding_front, padding_end)

    # lightweight coloring for motif and N's
    html = (
        template.replace("N", "<span style='background-color:red'>N</span>")
        .replace(motif, f"<span style='background-color:green'>{motif}</span>")
    )
    st.markdown(
        f"<div style='font-family:monospace; white-space:pre-wrap; word-break:break-word'>{html}</div>",
        unsafe_allow_html=True,
    )

with col3:
    st.subheader("Output")

    if run_button:
        # input sanity
        def ok_chars(s: str) -> bool:
            return all(ch in {"A", "U", "G", "C", "N"} for ch in s)

        if not (ok_chars(head) and ok_chars(motif) and ok_chars(tail)):
            st.error("Inputs must contain only characters A, U, G, C (and N for variable regions).")
        else:
            with st.spinner("Designing..."):
                result = staged_optimize_linear_rna(
                    motif=motif,
                    head=head,
                    tail=tail,
                    repetition_motif=repeats,
                    len_intermotifs=Ns_between,
                    padding_front=padding_front,
                    padding_end=padding_end,
                    iters=int(iters),
                )

            st.success("Design completed!")
            st.write(f"**Sequence length:** {len(result['sequence'])}")
            st.code(result["sequence"], language="text")
            st.text(f"MFE structure: {result['mfe_structure']}")
            st.text(f"MFE energy: {result['mfe_kcal_per_mol']:.2f} kcal/mol")
            st.text(f"Mean unpaired prob. (whole): {result['mean_pu_all']:.4f}")
            st.text(f"Linearized region: {result['linearize_seq']}")
            st.text(f"Mean unpaired prob. (linear): {result['mean_pu_linear']:.4f}")
            st.text(f"Min unpaired prob. (linear): {result['min_pu_linear']:.4f}")

            st.markdown("## 🧩 RNA Structure Visualization")
            # Recompute to generate images
            fc = RNA.fold_compound(result["sequence"])
            mfe_struct, _ = fc.mfe()
            fc.pf()
            centroid_struct, _ = fc.centroid()

            # Create SVG plots
            RNA.svg_rna_plot(result["sequence"], mfe_struct, "mfe_plot.svg")
            RNA.svg_rna_plot(result["sequence"], centroid_struct, "centroid_plot.svg")

            # Display
            st.markdown("### Minimum Free Energy (MFE) Structure")
            st.image("mfe_plot.svg")

            st.markdown("### Centroid Structure (Ensemble Average)")
            st.image("centroid_plot.svg")
            st.markdown("### Optimization Scores")
            st.line_chart(result["scores"])
