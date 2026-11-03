# Reddit Lens UI

Use the same quiet utility interface in the image lab, popup and Reddit overlay.

- **Type:** Google Sans, weights 400–700. Bundle the 35,852-byte Latin variable font locally; retain its OFL and source metadata in `public/fonts/`. No runtime font service requests. The Reddit overlay loads the same file with FontFace under a scoped family name, without changing Reddit's typography.
- **Color:** off-white `#f8f9fc`, white surfaces, dark ink `#272638`, muted text `#656579`. Lavender `#eeebfa` supports the purple action color `#6550ad`. Aquamarine `#e6f5f1` and deep teal `#14776f` communicate running/scanned states. Amber means uncertain; rose means failed. State labels always accompany color.
- **Layout:** upload and analysis first; model selection visible at the top. Image results are selectable thumbnails. Keep research details, component scores, exports and compression comparisons in expandable sections. Settings retains detector and scanning controls.
- **Popup:** status, model, scanning switches, optional source/badge guide, image lab button. Pair collection is disabled; do not expose collection controls.
- **Reddit:** keep the compact panel bottom right. Show status, model, scanned and flagged counts. Show failures and Retry when needed. Do not show pair-saving counters or successful collection messages in browsing badges. A scan failure must never look like a negative detection.
- **Interaction:** restrained hover/focus changes and a spinner only during actual work. Respect reduced motion. Preserve the existing detection, queueing, fetching and clipping behavior.
- **Copy:** short utility labels, sentence case, no marketing headings. Percentages are model scores, not probabilities or proof of origin.

References reviewed: [Impeccable](https://impeccable.style/) for hierarchy and removing repeated information; [Hallmark](https://github.com/nutlope/hallmark) for restraint and honest content; [getdesign.md](https://getdesign.md/) for a reusable visual system; [21st](https://21st.dev/) and [Minimal Gallery](https://minimal.gallery/) for interface references. No third-party component code or new UI dependencies were added. Refero did not return usable public content in this session.
