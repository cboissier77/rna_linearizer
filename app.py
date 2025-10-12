import RNA, random
import streamlit as st

def construct_rna_sequence(head, motif, tail, repetition_motif, len_intermotifs, padding_front=0, padding_end=0):
    """Construct the RNA sequence template with 'N' placeholders."""
    return head + padding_front * 'N' + (motif + len_intermotifs*'N')*(repetition_motif-1) + motif + padding_end * 'N' + tail

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
def optimize_linear_rna(motif, head, tail , repetition_motif, len_intermotifs, padding_front=0, padding_end=0, iters=5000, seed=0):
    random.seed(seed)
    construct = construct_rna_sequence(head, motif, tail, repetition_motif, len_intermotifs, padding_front, padding_end)
    linearize_start = len(head)
    linearize_end = len(construct) - len(tail) - 1
    # AU-biased random flanks
    def rand_base():
        return random.choices(["A","U","G","C"], weights=[0.25,0.25,0.25,0.25])[0]
    seq = "".join([rand_base() if c == 'N' else c for c in construct])

    def score(seq):
        pu = pu_vector(seq)
        pu_linear = pu[linearize_start:linearize_end+1]
        mean_pu = sum(pu_linear) / len(pu_linear)
        return mean_pu

    best = seq
    best_score = score(best)
    alphabet = ["A","U","G","C"]
    pos_of_N = [i for i, c in enumerate(construct) if c == 'N']

    # optimze for no head and tail first than add small percentage iteratively until full head and tail

    


    for _ in range(iters):
        pos = random.choice(pos_of_N)
        cand = best[:pos] + random.choice(alphabet) + best[pos+1:]
        sc = score(cand)
        if sc > best_score:
            best, best_score = cand, sc

    # Report MFE + accessibility
    fc = RNA.fold_compound(best)
    struct, mfe = fc.mfe()
    pu = pu_vector(best)
    return {
        "sequence": best,
        "mfe_structure": struct,
        "mfe_kcal_per_mol": mfe,
        "mean_pu_all": sum(pu)/len(pu),
        "linearize_seq": best[linearize_start:linearize_end+1],
        "mean_pu_linear": sum(pu[linearize_start:linearize_end+1])/(linearize_end-linearize_start+1),
        "min_pu_linear": min(pu[linearize_start:linearize_end+1])
    }

# ---------- Streamlit UI ----------
st.set_page_config(page_title="RNA Linear Region Designer", layout="wide")
st.title("🧬 RNA Linear Region Designer")

st.markdown("""
This tool designs an RNA sequence with a **linear (unpaired)** region of interest (ROI)
using the **ViennaRNA** thermodynamic model.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Input parameters")
    head = st.text_input("Head (fixed)", "AAA").upper()
    motif = st.text_input("Motif", "AAAGGG").upper()
    tail = st.text_input("Tail (fixed)", "UUU").upper()
    repeats = st.number_input("Motif repeats", 1, 10, 3)
    Ns_between = st.number_input("N between motifs", 0, 20, 5)
    padding_front = st.number_input("Padding front", 0, 100, 0)
    padding_end = st.number_input("Padding end", 0, 100, 0)
    iters = st.number_input("Iterations", 10, 500000, 400)
    seed = st.number_input("Random seed", 0, 9999, 1)
    run_button = st.button("🚀 Design RNA")
with col2:
    st.subheader("Constructed RNA template")
    construct = construct_rna_sequence(head, motif, tail, repeats, Ns_between, padding_front, padding_end)
    # text area comes back new line so not overflow horizontally, use different background color for motifs and Ns
    construct = construct.replace("N", "<span style='background-color: green;'>N</span>")
    construct = construct.replace(motif, "<span style='background-color: red;'>" + motif + "</span>")
    st.markdown(f"<div style='font-family: monospace; white-space: pre-wrap;'>{construct}</div>", unsafe_allow_html=True)

with col3:
    st.subheader("Output")

    if run_button:
        with st.spinner("Designing..."):
            result = optimize_linear_rna(
                motif=motif,
                head=head,
                tail=tail,
                repetition_motif=repeats,
                len_intermotifs=Ns_between,
                iters=iters,
                seed=seed
            )
            st.success("Design completed!")
            st.write(f"**Sequence length:** {len(result['sequence'])}")
            st.code(result['sequence'], language="text")
            st.text(f"MFE structure: {result['mfe_structure']}")
            st.text(f"MFE energy: {result['mfe_kcal_per_mol']:.2f} kcal/mol")
            st.text(f"Mean unpaired prob. (whole): {result['mean_pu_all']:.4f}")
            st.text(f"Linearized region: {result['linearize_seq']}")
            st.text(f"Mean unpaired prob. (linear): {result['mean_pu_linear']:.4f}")
            st.text(f"Min unpaired prob. (linear): {result['min_pu_linear']:.4f}")
        st.markdown("## 🧩 RNA Structure Visualization")

        fc = RNA.fold_compound(result['sequence'])
        mfe_struct, mfe_energy = fc.mfe()
        fc.pf()
        centroid_struct, _ = fc.centroid()
    

        # Create SVG plots
        RNA.svg_rna_plot(result['sequence'], mfe_struct, "mfe_plot.svg")
        RNA.svg_rna_plot(result['sequence'], centroid_struct, "centroid_plot.svg")

        # Display
        st.markdown("### Minimum Free Energy (MFE) Structure")
        st.image("mfe_plot.svg")

        st.markdown("### Centroid Structure (Ensemble Average)")
        st.image("centroid_plot.svg")

