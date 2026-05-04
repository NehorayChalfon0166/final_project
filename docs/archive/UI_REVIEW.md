# UI Review — Bitcoin Wallet Risk Analyzer

**Scope:** `Frontend/src/App.jsx`, `Frontend/src/pages/WalletAnalysis.jsx`, `Frontend/src/components/ErrorBoundary.jsx`, `Frontend/src/index.css` (1,499 lines), `Frontend/index.html`
**Skills applied:** `design:design-critique`, `design:accessibility-review` (WCAG 2.1 AA)
**Date:** 2026-05-04

---

## TL;DR

The product is a single-page React + Vite app with one screen (`WalletAnalysis`) that takes a Bitcoin address, calls a backend, and renders classification + balance + charts + transactions. The visual style is competent — Tailwind-style tokens, lucide icons, recharts — but two patterns are working against you:

1. **The actual answer (Criminal / Benign) is buried.** It sits *below* five graph-internal stats (Nodes, Edges, Neighbors, Risk Score, Confidence). The first thing the user came for is the fourth thing they see.
2. **Several interactive elements are `<div onClick>` instead of buttons,** which means keyboard users and screen readers can't use them — that's the biggest a11y gap.

Beyond that, contrast fails on a handful of light-gray text/icons and on the orange BTC value, the dark-mode trick of inverting the gray scale is clever but brittle, and three icon-only buttons need accessible names.

---

## 1. Design Critique

### Overall impression
Clean, modern fintech feel. Gradient navbar, card-based layout, sensible spacing (8px grid). The biggest opportunity is **result hierarchy** — restructure the result panel so the verdict leads, not the model internals.

### Usability

| Finding | Severity | Recommendation |
|---|---|---|
| Verdict ("Classification: Criminal/Benign") rendered *after* the stats grid (`WalletAnalysis.jsx:409` then `:508`). User scans Nodes/Edges/Neighbors before seeing the answer. | 🔴 Critical | Move the `classification-card` to the top of `results-container`, immediately after `result-header`. Make it the largest, most colorful element. |
| "Nodes / Edges / Neighbors" are model internals shown with equal weight to "Risk Score / Confidence". To a non-ML user they're noise. | 🟡 Moderate | Either collapse them under a "Model details" disclosure, or visually demote them (smaller cards, muted color, separate row). |
| Two `<h1>` on the page: nav has `<h1>Bitcoin Wallet Analyzer</h1>` (`App.jsx:27`) and page header has `<h1>Bitcoin Wallet Risk Analysis</h1>` (`WalletAnalysis.jsx:270`). The titles are also nearly identical. | 🟡 Moderate | Keep one `<h1>` on the page (the page header). Demote the navbar title to a `<p>` or `<span>` styled as a wordmark. |
| "Random Wallet" is a primary‑colored button, "History (N)" is secondary. They're at the same level of intent. | 🟢 Minor | Style both as secondary; the user's primary intent is the Analyze button above. |
| Risk score is displayed as a single percentage with no scale or context (`(risk_score * 100).toFixed(1)`). | 🟡 Moderate | Add a horizontal meter showing the score against the Low/Medium/High thresholds you already define in `getRiskLevel`. |
| "Powered by GNN Technology" footer (`App.jsx:42`) is jargon. | 🟢 Minor | "Powered by graph neural networks" or drop it. |
| Risk Score shows "%" but it's a probability, not a percentage of anything intuitive. | 🟢 Minor | Label as "Suspicion: 73 / 100" or "Risk: 73%". |
| `expandedTx` only allows one transaction expanded at a time. Comparing two transactions requires re-clicking. | 🟢 Minor | Allow multi-expand (track a `Set` of expanded txids). |
| Transaction list `max-height: 600px` with internal scroll inside the page (which already scrolls) creates a dual scroll trap. | 🟢 Minor | Drop the inner `max-height` and let the page scroll naturally; pagination already exists via "Load More". |

### Visual hierarchy
- **What draws the eye first:** the gradient navbar and big "Bitcoin Wallet Analyzer" — but the navbar repeats every page and shouldn't be the focal point of a result page. After analysis, the eye should land on the verdict.
- **Reading flow:** Title → Form → Stats grid (5 cards) → Balance → **Verdict** → Feature importance → Volume chart → Transactions. The verdict should be #3 not #5.
- **Emphasis:** Five equally-sized stat cards above the verdict create a flat hierarchy where everything looks equally important. Use size + color contrast to push the verdict up.

### Consistency

| Element | Issue | Recommendation |
|---|---|---|
| Color tokens | `--success-color: #10b981` defined, but charts use a different green `#22c55e` (`WalletAnalysis.jsx:641`, `index.css:1382`). Same for red: `--error-color: #ef4444` matches charts but `classification-card.green` uses `#10b981`. | Pick one green and one red, reference the token everywhere. |
| Gray scale | Tailwind grays are inverted in dark mode by **swapping every `--gray-*` value** (`index.css:30-47`). So `var(--gray-200)` for a border in light = `#e5e7eb`, in dark = `#4b5563` — works. But `color: var(--gray-700)` in light is dark text; in dark it becomes pure-light text — also works *by accident* for color but breaks any rule that wanted a "subtle gray" semantically. | Replace the inversion trick with semantic tokens (`--text-muted`, `--surface`, `--border-subtle`) defined per theme. The current pattern is hard to maintain. |
| BTC color | Orange `#f59e0b` is used for: `chart-header` icon (`index.css:1338`), main balance value (`:1476`), accent border on main balance card. Nice, but BTC orange is *also* the `--warning-color`, so warning alerts and BTC value share a color. | Use a separate `--btc-orange` token; reserve `--warning-color` for warnings only. |
| Spacing | Card padding is mostly `1.5rem`, but `transaction-summary` uses `1rem 1.5rem` (`:1128`) and `balance-item` uses `1rem` (`:1410`). | Standardize on `1.5rem` card padding or document the system. |
| `.btn-secondary` text color is `var(--gray-700)` (= `#374151`) in light but in dark becomes `#f9fafb` — fine. But the button background `var(--gray-200)` becomes `#4b5563` in dark, against light text that's ~7:1 — passes, but it visually competes with primary buttons in dark mode. | Define a true secondary token decoupled from gray-200. |
| Icon-only header in `chart-header` (`:1338`) is hard-coded orange `#f59e0b` regardless of chart type. Both Feature Importance and Volume charts get an orange icon; volume's bars are green/red. | Either match the chart's primary hue or use a neutral icon color. |

### What works well
- Token-driven CSS variables make theming cheap.
- Dark mode is implemented and persisted in `localStorage`.
- ErrorBoundary catches render errors and offers a reload — good defensive UX.
- Empty/loading states exist for the form, feature-importance card, and chart.
- Export buttons (JSON, PDF) are nicely placed in the result header.
- Mobile breakpoint exists at 768px and stacks the input/button column.
- Search history with classification badges is a thoughtful touch.

### Priority recommendations
1. **Lead with the verdict.** Move `classification-card` to the top of `results-container`, make it the visual anchor, and demote the model internals.
2. **Replace the inverted-gray dark-mode hack with semantic tokens.** It's a small refactor that will save a lot of one-off `.dark-mode .x` overrides (you already have ~20 of them in `index.css:49-156`).
3. **Make every clickable thing a real `<button>`** — this fixes the biggest a11y problem (next section) and removes a class of bugs.

---

## 2. Accessibility Audit (WCAG 2.1 AA)

**Issues found:** 14 | **Critical:** 4 | **Major:** 6 | **Minor:** 4

### Perceivable

| # | Issue | WCAG | Severity | Recommendation |
|---|---|---|---|---|
| 1 | `--gray-400` (#9ca3af) used as text on white in `.copy-btn` (`:647`), `.external-link` (`:1157`), `.expand-btn` (`:1195`), `.toggle-btn` (`:1109`), `.tx-io-item .op-return` (`:1265`), `.balance-count` (`:1487`), `.more-items` (`:1271`). Contrast ≈ 2.85:1. | 1.4.3 | 🔴 Critical | Use `--gray-500` (#6b7280, ratio 4.83:1) or darker. |
| 2 | `--primary-color` (#3b82f6) as text on white in `.tx-id code` (`:1153`) and `.more-transactions a` (`:1306`). Contrast ≈ 3.68:1. | 1.4.3 | 🟡 Major | Use `--primary-dark` (#2563eb, 5.17:1) for text-on-white. |
| 3 | `.balance-value.btc { color: #f59e0b }` on white card (`:1476`). Contrast ≈ 2.16:1 even at 24px/600 (still fails 3:1 large-text). | 1.4.3 | 🔴 Critical | Use `#b45309` (amber-700) for the BTC value text on light backgrounds, or render the icon in orange and the value in `--text-primary`. |
| 4 | `.form-input::placeholder` is the browser default light-gray on white. | 1.4.3 | 🟡 Major | Set `::placeholder { color: var(--gray-500) }` explicitly. |
| 5 | Recharts axis ticks hard-coded to `fill: '#6b7280'` (`WalletAnalysis.jsx:560,567,613,617`). On white that's 4.83:1 (pass). In **dark mode** they remain `#6b7280` against `#1f2937` panel ≈ 3.5:1 — fails for normal text. | 1.4.3 | 🟡 Major | Use a CSS variable read via `getComputedStyle`, or pass different tick colors in dark mode. |
| 6 | Recharts Tooltip background is hard-coded `'white'` (`:573, :624`). In dark mode it's a stark white pop on a dark canvas — readable but jarring. | 1.4.11 | 🟢 Minor | Theme the tooltip with the surface color. |
| 7 | Status / classification rely on **color + icon + text** — text and icon are present, so this passes 1.4.1 (use of color). ✅ | 1.4.1 | — | No action. |
| 8 | Decorative emoji `😵` in `<h1>` of ErrorBoundary (`ErrorBoundary.jsx:22`) — screen readers will announce "dizzy face". | 1.1.1 | 🟢 Minor | Wrap in `<span aria-hidden="true">` or replace with a lucide icon. |

### Operable

| # | Issue | WCAG | Severity | Recommendation |
|---|---|---|---|---|
| 9 | `<div onClick>` on three interactive surfaces, none keyboard-reachable: `transactions-header` (`:675`), `transaction-summary` (`:692`), `history-item` (`:347`). | 2.1.1 | 🔴 Critical | Convert to `<button type="button">` with `display: flex; width: 100%; text-align: left; background: none; border: none;`. The expand/collapse rows in particular block keyboard users entirely. |
| 10 | The collapse chevron inside `transactions-header` is a `<button>` (`:682`) but has no `onClick` — clicks bubble to the parent div. Tab focus lands on the chevron and Enter does nothing. | 2.1.1, 4.1.2 | 🟡 Major | Either remove the inner button (header itself becomes the button) or wire the button's onClick. |
| 11 | `.form-input:focus { outline: none }` (`:621`) replaces outline with a 2px primary-color border. Border `#3b82f6` on white is 3.68:1 against white — passes 1.4.11 (3:1) but only just. No focus styles defined for `.btn`, `.example-btn`, `.copy-btn`, `.toggle-btn`, `.theme-toggle-nav`, `.expand-btn`, etc. | 2.4.7 | 🟡 Major | Add a global `:focus-visible { outline: 2px solid var(--primary-dark); outline-offset: 2px; }` rule. |
| 12 | Touch targets too small: `.copy-btn` icon 18px + 0.25rem padding ≈ 24px (`:642`); `.toggle-btn` ≈ 24px (`:1106`); `.expand-btn` ≈ 24px (`:1188`); `.external-link` icon-only. WCAG 2.1 AA recommends 44×44 (2.5.5 is AAA in 2.1, AA in 2.2 at 24×24). | 2.5.5 | 🟡 Major | Pad icon buttons to at least 44×44, or 24×24 minimum and ensure spacing between them. |
| 13 | Tab order is mostly DOM order which is fine, but `expandedTx` and `showTransactions` toggles have no `aria-expanded` / `aria-controls`. | 4.1.2 | 🟡 Major | Add `aria-expanded={showTransactions}` and `aria-controls="tx-list"` on the toggle. |

### Understandable

| # | Issue | WCAG | Severity | Recommendation |
|---|---|---|---|---|
| 14 | Error alert (`<div className="alert alert-error">`, `WalletAnalysis.jsx:372`) is rendered without `role="alert"` or `aria-live="polite"`. Screen reader users won't be notified when the analysis fails. | 4.1.3 / 3.3.1 | 🟡 Major | Add `role="alert"` (or wrap in an `aria-live="assertive"` region). |
| 15 | The form input has a label (`htmlFor="wallet-address"`) ✅ but no `aria-describedby` pointing at the help text "Get a wallet from latest transactions:". Also no input format hint (Bech32 vs P2PKH addresses). | 3.3.2 | 🟢 Minor | Add `aria-describedby="address-hint"` and an `id` on the helper text. Consider a `pattern` or live validation. |

### Robust

| # | Issue | WCAG | Severity | Recommendation |
|---|---|---|---|---|
| 16 | Icon-only buttons lack accessible names: theme toggle (`App.jsx:29` — has `title` only), copy button (`:290` — `title` only), external-link anchors (`:698` — no aria-label), expand chevrons. `title` attributes are not announced reliably. | 4.1.2 | 🔴 Critical | Add `aria-label="Switch to dark mode"`, `aria-label="Copy address"`, `aria-label="View on mempool.space"`, `aria-label="Expand transaction details"`. |
| 17 | Charts are `<svg>` from recharts with no `<title>` / `aria-label` / `role="img"`. | 1.1.1 | 🟡 Major | Wrap each chart in `<div role="img" aria-label="Bar chart of weekly received vs. sent BTC. Total received X BTC, total sent Y BTC.">`. |

### Color contrast spot check (light theme on white)

| Element | Color | Ratio | Required | Pass |
|---|---|---|---|---|
| Body text (`--gray-900` #111827) | on `--gray-50` #f9fafb | 18.7:1 | 4.5 | ✅ |
| Page subtitle (`--gray-600` #4b5563) | on white | 7.56:1 | 4.5 | ✅ |
| Muted text (`--gray-500` #6b7280) | on white | 4.83:1 | 4.5 | ✅ |
| `--gray-400` icons & `op-return` text | on white | 2.85:1 | 4.5 / 3.0 | ❌ |
| Primary blue link `#3b82f6` | on white | 3.68:1 | 4.5 | ❌ |
| BTC orange `#f59e0b` (24px/600) | on white | 2.16:1 | 3.0 | ❌ |
| Footer `#d1d5db` | on `#1f2937` | 8.35:1 | 4.5 | ✅ |
| Alert text `#991b1b` | on `#fee2e2` | 7.66:1 | 4.5 | ✅ |
| `status-badge.success` `#065f46` | on `#d1fae5` | 7.32:1 | 4.5 | ✅ |

### Keyboard navigation summary

| Element | Tab reachable | Enter/Space | Notes |
|---|---|---|---|
| Address input | ✅ | submits form | OK |
| Analyze button | ✅ | submits | OK |
| Copy button | ✅ | copies | Missing accessible name |
| Random Wallet | ✅ | works | OK |
| History toggle | ✅ | toggles | OK |
| **History item** | ❌ | — | `<div onClick>` — broken |
| Theme toggle | ✅ | toggles | Missing accessible name |
| **Transactions header** | ❌ | — | `<div onClick>` — broken |
| **Transaction summary row** | ❌ | — | `<div onClick>` — broken |
| Toggle chevron (in tx header) | ✅ | nothing | Has no onClick |
| External link icon | ✅ | opens link | Missing accessible name |
| Export buttons | ✅ | works | OK |

---

## 3. Priority Fix List (in order)

1. **Convert all `<div onClick>` interactive rows to `<button>`** — `transactions-header`, `transaction-summary`, `history-item`. (Critical accessibility, ~30 min.)
2. **Add `aria-label` to every icon-only button**: theme toggle, copy, external links, expand chevrons. (Critical accessibility, ~15 min.)
3. **Move the verdict above the stats grid** — restructure `results-container` so `classification-card` renders first and is the visual anchor. (Critical UX, ~10 min.)
4. **Fix the contrast failures**: replace `var(--gray-400)` text-on-white with `--gray-500` or darker; use `--primary-dark` for blue text; use a darker amber for BTC value text. (~20 min.)
5. **Add a global `:focus-visible` style** so keyboard users can see where they are. (~5 min.)
6. **Add `role="alert"` to the error alert and `aria-expanded` to the collapsibles.** (~10 min.)
7. **Demote the model internals (Nodes/Edges/Neighbors)** to a smaller, secondary row or a "Model details" disclosure. (~15 min.)
8. **Refactor dark mode** from "invert the grays" to semantic tokens (`--surface`, `--text-muted`, `--border`). (Half-day refactor.)
9. **Theme recharts** for dark mode (axis ticks + tooltip surface). (~30 min.)

---

## 4. Quick wins as code

```jsx
// App.jsx — accessible theme toggle
<button
  className="theme-toggle-nav"
  onClick={() => setDarkMode(!darkMode)}
  aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
>
  {darkMode ? <Sun size={20} aria-hidden="true" /> : <Moon size={20} aria-hidden="true" />}
</button>
```

```jsx
// WalletAnalysis.jsx — accessible transactions header
<button
  type="button"
  className="transactions-header"
  onClick={() => setShowTransactions(!showTransactions)}
  aria-expanded={showTransactions}
  aria-controls="tx-list"
>
  <span className="transactions-title">
    <ArrowRightLeft size={24} aria-hidden="true" />
    <h3>Transactions ({walletInfo.transaction_count})</h3>
  </span>
  {showTransactions ? <ChevronUp size={20} aria-hidden="true" /> : <ChevronDown size={20} aria-hidden="true" />}
</button>
```

```css
/* index.css — global focus + fix the worst contrast failures */
:focus-visible {
  outline: 2px solid var(--primary-dark);
  outline-offset: 2px;
  border-radius: 4px;
}
.form-input::placeholder { color: var(--gray-500); }
.copy-btn,
.external-link,
.expand-btn,
.toggle-btn,
.balance-count,
.more-items,
.tx-io-item .op-return { color: var(--gray-500); }
.tx-id code,
.more-transactions a { color: var(--primary-dark); }
.balance-value.btc { color: #b45309; } /* amber-700 */
```
