---
version: 1
slug: "app-home-py"
primary_target: "app/Home.py"
related_targets: ["app/pages"]
---

Scope: the whole Streamlit app (app/Home.py and app/pages/*). Visitor mode: Operate.
Audience: viva examiners on a projector or laptop, a few minutes of clicking. Job: price a car, see why, verify model quality.
Constraints: fixed Streamlit stack; prices in ₹ Indian grouping plus lakh; every number traceable to reports/tables or model metadata; light and dark modes; contrast >= 4.5:1.
Craft bar (user-named): Linear, Vercel dashboard, Stripe dashboard.

## Direction contract

THESIS: The category standard (a clean product dashboard) played straight at Linear/Vercel/Stripe craft. It refuses the template look of a default Streamlit app: equal cards everywhere, a weak hierarchy, and decorative colour.

OWN-WORLD: A zinc neutral ramp (page #FAFAFA, surface #FFFFFF, border #E4E4E7, ink #18181B, muted #71717A), one blue accent #2F5BEA for actions, selection and the main chart series, and amber #D97706 as the single highlight per view. Geist for UI and headings, Geist Mono with tabular figures for data. 1px borders, 8px card radius, 6px controls, no shadows except on overlays. Dark mode mirrors it on #0B0B0D.

STORY: The examiner opens the estimator, sees a priced car straight away, changes an input, re-estimates, reads why, then checks Model performance and the About page for evidence.

FIRST VIEWPORT: A sticky header with a wordmark and top tabs. H1 and a one-line lede. On the left (40%), the input panel with Vehicle, Condition, and Engine and fuel groups, and a full-width primary button. On the right (60%), the result panel is the dominant region: a muted label, the estimate in large Geist semibold tabular figures, a lakh and 80% range line, the range bar with an amber marker, a note on similar listings, and a provenance line.

FORM: canon (standing exit chosen on the decision page); seed:37032769f1568db8. Signature move: the range bar with one amber estimate marker, echoed as the single amber highlight on each page that ranks or compares data (Market explorer, Model performance); text-and-table pages such as About and data carry no amber.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
