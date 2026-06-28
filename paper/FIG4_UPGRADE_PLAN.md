# Fig. 4 Upgrade Plan: 2-curve → 3-curve

**Trigger:** ISIC 2018 gate=pam window ablation (M∈{7,28,56,112}) results available  
**File to edit:** `paper/sections/06_analysis.tex` (lines 36–82)

---

## Placeholders to fill first

Compute from ISIC gate=pam test results vs DA-TransUNet baseline (88.88%):

| Placeholder | Formula | Source run |
|-------------|---------|------------|
| `[A]` | ISIC gate=pam M=7 DSC − 88.88 | `isic_pam_M7_300ep.log` |
| `[B]` | ISIC gate=pam M=28 DSC − 88.88 | `isic_pam_M28_300ep.log` |
| `[C]` | ISIC gate=pam M=56 DSC − 88.88 | `isic_pam_M56_300ep.log` |
| `[D]` | ISIC gate=pam M=112 DSC − 88.88 | `isic_pam_M112_300ep.log` |
| `[SPAN]` | max(A,B,C,D) − min(A,B,C,D) | computed |
| `[ISIC_BEST]` | max(A,B,C,D), formatted as +X.XX | computed |
| `[ISIC_M]` | M achieving max DSC | computed |

If any ISIC point exceeds `ymax=3.5` or drops below `ymin=-1.75`, adjust those axis limits.

---

## 1. Replace TikZ figure block (lines 36–82)

```latex
\begin{figure}[ht]
\centering
\scalebox{1.0}{%
\begin{tikzpicture}
\begin{axis}[
  width=\columnwidth, height=5.3cm,
  xlabel={Window size $M$},
  ylabel={$\Delta$DSC vs.\ DA-TransUNet (\%)},
  ymin=-1.75, ymax=3.5,
  xtick={7,28,56,112},
  xticklabels={7,28,56,112},
  xmin=4, xmax=120,
  ymajorgrids=true,
  grid style={dashed,gray!30},
  tick label style={font=\footnotesize},
  label style={font=\footnotesize},
  every axis plot/.append style={very thick, mark size=2.5pt},
  legend style={at={(0.02,0.02)}, anchor=south west, font=\scriptsize,
                draw=gray!50, fill=white, inner sep=3pt},
]
% Zero reference
\addplot[dashed, gray!50, very thin, forget plot] coordinates {(4,0)(120,0)};
% Synapse
\addplot[color=red!70!black, mark=*]
  coordinates {(7,-1.16) (28,1.14) (56,-0.90) (112,-0.36)};
\addlegendentry{Synapse}
% ISIC 2018
\addplot[color=blue!65, mark=triangle*]
  coordinates {(7,[A]) (28,[B]) (56,[C]) (112,[D])};
\addlegendentry{ISIC 2018}
% Kvasir-SEG
\addplot[color=green!50!black, mark=square*]
  coordinates {(7,1.09) (28,1.10) (56,1.73) (112,1.12)};
\addlegendentry{Kvasir-SEG}
% Peak label — Synapse
\node[font=\scriptsize, color=red!80, anchor=south]
  at (axis cs:28,1.14) {$+$1.14\%};
% Peak label — Kvasir
\node[font=\scriptsize, color=green!60!black, anchor=south]
  at (axis cs:56,1.73) {$+$1.73\%};
% Span annotation box — no GCS labels, no ratio
\node[anchor=north east, align=left, font=\scriptsize,
      fill=white, draw=gray!50, inner sep=3pt, rounded corners=2pt]
  at (axis cs:119, 3.40) {%
    \textcolor{red!70!black}{Synapse span:\phantom{0} 2.30\,pp}\\
    \textcolor{blue!65}{ISIC span:\phantom{000} [SPAN]\,pp}\\
    \textcolor{green!50!black}{Kvasir span:\phantom{00} 0.64\,pp}};
\end{axis}
\end{tikzpicture}}
\caption{$\Delta$DSC vs.\ window size $M$ (gate=pam, $r{=}32$, 300ep) for three
representative medical image segmentation tasks. The three tasks exhibit
qualitatively different sensitivities to approximation strength: Synapse
multi-organ CT shows a non-monotonic curve with a 2.30\,pp span and a
clear optimum at $M{=}28$; ISIC~2018 and Kvasir-SEG show near-flat curves
([SPAN]\,pp and 0.64\,pp respectively), indicating these tasks are largely
window-robust.}
\label{fig:dsc_vs_m}
\end{figure}
```

---

## 2. Body text update (§6.2, ~line 91)

**Replace** the current single ISIC sentence:
```
On ISIC 2018, ApproxDA-TransUNet (gate=learn, $M{=}7$) achieves 89.58\% DSC
vs.\ 88.88\% for DA-TransUNet ($+$0.70\%), consistent with the low-sensitivity pattern.
```

**With:**
```
On ISIC 2018, the gate=pam window ablation (Figure~\ref{fig:dsc_vs_m}) confirms the
same low-sensitivity pattern: the DSC span across $M\in\{7,28,56,112\}$ is [SPAN]\,pp,
consistent with Kvasir-SEG and substantially lower than Synapse.
```

**Also update** the gains list sentence to use gate=pam ISIC peak:
```
% Old:
gains are positive across all three benchmarks: $+$1.14\% (Synapse, $M{=}28$),
$+$1.73\% (Kvasir, $M{=}56$), and $+$0.70\% (ISIC, $M{=}7$).

% New:
gains are positive across all three benchmarks: $+$1.14\% (Synapse, $M{=}28$),
$+$1.73\% (Kvasir, $M{=}56$), and $+$[ISIC_BEST]\% (ISIC, $M{=}$[ISIC_M]).
```

---

## Design notes

- **No GCS terminology** in caption or annotation box — do not write "high GCS" / "low GCS" in Fig. 4 (GCS is formally introduced later in §6.3).
- **No ratio** — replace "3.6×" with per-dataset spans; the three individual numbers are more informative and stay self-consistent when a third dataset is added.
- **Curve color**: ISIC = `blue!65`, mark = `triangle*` (distinct from red circle and green square).
- **y-axis range**: `[-1.75, 3.5]` should accommodate ISIC if it follows the expected low-sensitivity pattern (approx. −0.5 to +1.0 range). Adjust if needed.
- **Peak label for ISIC**: Add a `\node` for the ISIC peak analogous to the Synapse and Kvasir labels, once the best M is known.
