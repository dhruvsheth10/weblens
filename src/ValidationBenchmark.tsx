import {useState} from 'react';
import {VALIDATION_NOTE, VALIDATION_ROWS, type ValidationScope, type ValidationVariant} from './detection/validationBenchmark';

const VARIANTS: ValidationVariant[] = ['original', 'reddit-webp640', 'reddit-webp960'];
const SCOPES: ValidationScope[] = ['full-folder', 'new-unique-bytes'];
const VARIANT_LABEL: Record<ValidationVariant, string> = {
  original: 'Original',
  'reddit-webp640': 'WebP 640',
  'reddit-webp960': 'WebP 960',
};
const SCOPE_LABEL: Record<ValidationScope, string> = {
  'full-folder': 'Full folder',
  'new-unique-bytes': 'New unique bytes',
};

const percent = (value: number | null) => value === null ? '—' : `${(value * 100).toFixed(1)}%`;

export function ValidationBenchmark({activeModel}: {activeModel: string}) {
  const [variant, setVariant] = useState<ValidationVariant>('original');
  const [scope, setScope] = useState<ValidationScope>('full-folder');
  const rows = VALIDATION_ROWS.filter(row => row.variant === variant && row.scope === scope)
    .sort((a, b) => b.partialAuRoc01 - a.partialAuRoc01 || a.name.localeCompare(b.name));
  const sample = rows[0];
  const sherlock = rows.find(row => row.model === 'reddit-sherlock-2');
  return <section className="published-benchmark" aria-labelledby="validation-heading">
    <span className="eyebrow">17 Oct 2026 · packaged cutoffs</span>
    <h2 id="validation-heading">Stored models on the validation folder</h2>
    <p>Nine local models scored every supported file at the cutoff already packaged with that model. FiSeR was stopped before it finished. Nothing here was used to pick a threshold.</p>
    <div className="benchmark-switch" role="group" aria-label="Image version">{VARIANTS.map(item => <button key={item} type="button" aria-pressed={item === variant} onClick={() => setVariant(item)}>{VARIANT_LABEL[item]}</button>)}</div>
    <div className="benchmark-switch" role="group" aria-label="Which files to count">{SCOPES.map(item => <button key={item} type="button" aria-pressed={item === scope} onClick={() => setScope(item)}>{SCOPE_LABEL[item]}</button>)}</div>
    {sample && sherlock && <div className="benchmark-stats">
      <div><span>Scored images</span><strong>{sample.n.toLocaleString()}</strong></div>
      <div><span>AI / real</span><strong>{sample.ai.toLocaleString()} / {sample.real.toLocaleString()}</strong></div>
      <div><span>Sherlock 2 AI recall</span><strong>{percent(sherlock.recall)}</strong></div>
      <div><span>Sherlock 2 false flags</span><strong>{sherlock.fp} of {sherlock.real}</strong></div>
    </div>}
    <div className="table-scroll"><table className="published-table">
      <thead><tr><th>Model</th><th>Cutoff</th><th>AI recall</th><th>False flags</th><th>Precision</th><th>AUROC</th><th>Partial AUROC</th><th>Median</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.model} className={row.model === activeModel ? 'is-selected' : undefined}>
        <td>{row.name}</td>
        <td className="mono">{row.cutoff.toFixed(3)}</td>
        <td className="mono">{percent(row.recall)}</td>
        <td className="mono">{row.fp} · {percent(row.fpr)}</td>
        <td className="mono">{percent(row.precision)}</td>
        <td className="mono">{row.auroc.toFixed(3)}</td>
        <td className="mono">{row.partialAuRoc01.toFixed(3)}</td>
        <td className="mono">{Math.round(row.medianMs)} ms</td>
      </tr>)}</tbody>
    </table></div>
    <p className="subtle">Sorted by standardized partial AUROC with the false-flag rate capped at 1%. The highlighted row is the model selected in Settings. {VALIDATION_NOTE.decodeFailures} of {VALIDATION_NOTE.attempted.toLocaleString()} files failed to decode and are left out of every count. {VALIDATION_NOTE.calibrationOverlap.toLocaleString()} scored files also appear in calibration, and {VALIDATION_NOTE.overlapAi.toLocaleString()} of those are AI, so full-folder recall is not an independent test. The byte-new slice has {VALIDATION_NOTE.newUniqueAi} AI images, so its recall can swing on a few files. {VALIDATION_NOTE.newUniqueReal} of the real images are byte-new, so the false-flag rate barely changes between the two scopes.</p>
  </section>;
}
