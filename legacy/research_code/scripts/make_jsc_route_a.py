#!/usr/bin/env python3
"""Derive the three-experiment manuscript in an explicitly supplied checkout."""
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve(); p=root/'LaTeX_DL_468198_240419/template.tex';s=p.read_text()
def between(a,b,value):
 global s
 i=s.index(a);j=s.index(b,i);s=s[:i]+value+s[j:]
between(r'\subsection{Experiment 3: Photonic-crystal',r'\subsection{Experiment 5: Three-dimensional', '')
s=s.replace(r'\subsection{Experiment 5:',r'\subsection{Experiment 3:')
between('Across five two- and three-dimensional experiments',r'\keywords',r'''Across three two- and three-dimensional experiments, DD reduces stored basis
entries by \(31.8\%\)--\(48.3\%\). A half-space storage--time and
training-coverage study includes a competitive fixed solution-POD initial guess
with the same completion and true residual target. It identifies beneficial
DD choices at comparable trajectory-model storage budgets, while simpler reuse
remains more economical in this parameter family. On a
\(680{,}225\)-unknown three-dimensional waveguide, DD reduces accounted ROM
payload by \(34.7\%\) and retains about a factor-two speedup over
shifted-Laplacian multigrid GMRES, while taking longer than Global TA-ROM.
Online inverse-action defects and reduced-system stability checks illustrate
the conditional residual bound and the problem dependence of storage allocation.
''')
s=s.replace('Experiments 1--3 this may','Experiments 1--2 this may').replace('Experiments 4--5 instead','Experiment 3 instead').replace('used for Experiments 1--3','used for Experiments 1--2')
s=s.replace('DD-TA-ROM behavior, a photonic-crystal waveguide test, and two large\nthree-dimensional PML problems.', 'DD-TA-ROM behavior, and a large three-dimensional PML waveguide.')
s=s.replace('We evaluate DD-TA-ROM on five parameterized Helmholtz problems: a two-material\nhalf-space, elliptic-obstacle scattering, a two-dimensional photonic-crystal\nwaveguide, a three-dimensional Gaussian lens, and a three-dimensional Gaussian\nwaveguide.', 'We evaluate DD-TA-ROM on three parameterized Helmholtz problems: a two-material\nhalf-space, elliptic-obstacle scattering, and a three-dimensional Gaussian\nwaveguide.')
s=s.replace('The first three problems use three reduced layers, and the two large\nthree-dimensional problems use two reduced layers.', 'The two-dimensional problems use three reduced layers, and the\nthree-dimensional waveguide uses two reduced layers.')
s=s.replace('  The photonic-crystal training references are verified direct Helmholtz solutions.', '')
s=s.replace('DD-TA-ROM behavior, and a large three-dimensional PML waveguide.  Section~\\ref{sec:conclusion}\nconcludes, and Appendix~\\ref{app:waveholtz-matching} records the discrete\nWaveHoltz matching details for the PML operator.', 'DD-TA-ROM behavior, and a three-dimensional PML waveguide.\nSection~\\ref{sec:conclusion} concludes the paper.\nAppendix~\\ref{app:waveholtz-matching} gives the discrete PML matching details.')
s=s.replace('reports half-space trajectory evidence, elliptic-obstacle\nDD-TA-ROM behavior, and a three-dimensional PML waveguide.','reports half-space and scattering tests, followed by a large\nthree-dimensional PML waveguide.')
s=s.replace('The second large test uses' , 'The three-dimensional test uses').replace('again with width','with width')
between(r'\((0.0056,0.0278)\); the snapshot and online solver settings',r'Figure~\ref{fig:3d-waveguide-geometry}',r'''\((0.0056,0.0278)\). Training requests an inverse-action relative residual
of \(10^{-7}\), using restart 40 and at most ten cycles. The actual recorded
residuals exceed the request in 7 of 12 Global and 2 of 12 DD snapshots,
with maxima \(3.19\times10^{-7}\) and \(2.56\times10^{-7}\). These achieved
values characterize the archived dataset; final online residuals are
independently accepted at \(10^{-6}\). New strict runs reject snapshots
that miss their target, and their timings are not mixed into these tables.
Online shifted-Laplacian MG-GMRES uses restart 40, at most 20 cycles, and
complex shift \(\beta=1.2\). Each application is a six-level V-cycle with
two pre- and two post-Jacobi sweeps, weight 0.6, and at most 3000 coarse DOFs.
Five physical Gaussian-profile bands, separated at
\(10^{-3},10^{-2},10^{-1},0.5\), are each divided into five axial sections.
PML cells use signed-face ownership, split at half the depth; edge/corner
ownership follows a fixed \(x\)-\(y\)-\(z\) priority convention.
''')
s=s.replace(r'\paragraph{Joint three-dimensional online-cost comparison.}',r'\paragraph{Online cost.}')
s=s.replace('These are single full-solve measurements. The new lens repetition study is\nreported separately in Table~\\ref{tab:jsc-lens-tradeoff}; its short reduced\nphases are not compared using a mixture of timing cohorts.', 'These are single full-solve measurements; the short half-space reduced phases\nare supplemented by the separate five-repetition reuse study.')
between('The maps occupy 5.024 MiB for the lens',r'\begin{table}[t]',r'''The map occupies 5.190 MiB. DD reduces basis storage by \(41.0\%\) and
accounted payload by \(34.7\%\). Its modes are stored only on their owning
blocks; total reduced rank alone is therefore not an appropriate memory metric.

''')
a=s.index('Lens & Global &');b=s.index('Waveguide & Global &',a);s=s[:a]+s[b:]
between('Single Global and DD offline construction measurements are',r'\begin{table}[t]',r'''Single Global and DD offline construction measurements are 268.6 and 369.4
seconds using four MATLAB process workers. Against the reported baseline,
DD amortizes this cost after approximately 19--20 queries. The MG hierarchy
occupies another 260.9 MiB, required for completion in this example. Full-order
assembly, Krylov workspaces, and process overhead are additional to ROM
payload; no complete-process peak-memory saving is inferred from the latter.

A separate sparse-direct measurement at the first query takes 139.26 seconds
for factorization and 3.16 seconds for solution, with a 45.8 GB process RSS
peak. This is a scale reference rather than a repeated timing statistic.

''')
s='\n'.join(line for line in s.split('\n') if not line.startswith('Lens & 200.53'))
between('The three-dimensional experiments test a competitive',r'\subsection{Controlled audits',r'''The three-dimensional experiment tests a competitive shifted-MG completion
at a common true residual target of \(10^{-6}\). Across the three primary
configurations, DD basis savings are \(31.8\%\), \(48.3\%\), and
\(41.0\%\). These values concern stored basis entries; the accompanying
tables also include reduced operators and DD indexing. Storage and total
online cost must be considered together when choosing a representation.

''')
s=s.replace('The photonic-crystal references\nare direct Helmholtz solutions.  ', '')
s=s.replace('The large three-dimensional\nexperiments supply', 'The three-dimensional\nexperiment supplies')
between('The frozen-model tests in Tables~',r'\input{revision_audit_tables.tex}',r'''The frozen-model waveguide tests in Table~\ref{tab:frozen-holdout} use the
archived draw with random seed 9052026: three new interior queries and one query 0.02
beyond each training endpoint, without changing ranks. All methods pass the
true \(10^{-6}\) residual test; the upper extrapolation point reduces DD's
speedup against MG to 1.50. These small one-parameter checks complement the
half-space coverage study without establishing unrestricted extrapolation
or frequency robustness.

\input{jsc_action_diagnostics.tex}

''')
between('DD allows local storage allocation, but its usefulness depends', 'The main limitations are',r'''DD allows local storage allocation, but its usefulness depends on the complete
storage--time tradeoff. At comparable half-space ROM budgets, DD can improve
total time over a Global trajectory model; a tightly retained fixed solution
POD is more economical in this family. On the \(680{,}225\)-unknown waveguide,
DD saves \(34.7\%\) of accounted ROM payload and remains about twice as fast
as shifted-MG alone, while taking longer than Global. Full process memory
also includes the full-order operators and completion solver.

''')
s=s.replace('reach angular frequencies 24\nand 32 and at most', 'include angular frequency 24\nand at most').replace('five experiments','three experiments').replace('five test','three test');p.write_text(s)
p=root/'LaTeX_DL_468198_240419/revision_audit_tables.tex';s=p.read_text();s='\n'.join(l for l in s.split('\n') if not l.startswith('Photonic crystal &'));s=s.replace('five-repetition means for PC and archived single measurements for the other families','archived single measurements for both families');p.write_text(s)
p=root/'LaTeX_DL_468198_240419/jsc_action_diagnostics.tex';s=p.read_text();a=s.index('The loose lens DD');b=s.index('For the six-snapshot',a);s=s[:a]+s[b:];s='\n'.join(l for l in s.split('\n') if not l.startswith('Lens,'));s=s.replace('Queries are 0.785 for half-space, 0.283 for the lens, and 0.317 for the waveguide.','Queries are 0.785 for half-space and 0.317 for the waveguide.');p.write_text(s)
assert r'\subsection{Experiment 3: Three-dimensional' in (root/'LaTeX_DL_468198_240419/template.tex').read_text()
print('Route A manuscript derived at',root)
