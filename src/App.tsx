import {useEffect, useRef, useState, type DragEvent} from 'react';
import {ArrowDownToLine, ArrowRight, ArrowUpRight, ChevronRight, FlaskConical, ImagePlus, Layers3, LoaderCircle, ScanLine, Settings2, ShieldCheck, SlidersHorizontal, X} from 'lucide-react';
import {analyze, resetAnalyzer} from './detection/client';
import {MODELS, isRedditStudent, modelConfig, verdict, decisionBand, variantLabel, type Detection, type ModelId, type Variant, type BenchmarkRow} from './detection/model';
import {metrics} from './detection/metrics';
import {ScanSettings} from './ScanSettings';
import {getSettings, DEFAULT_SETTINGS, sourceForModelSwitch, type Settings} from './extension/settings';
import {ValidationBenchmark} from './ValidationBenchmark';

const isExtension = location.protocol === 'chrome-extension:';
type Tab = 'inspect' | 'benchmark' | 'settings';
type LabImage = {id: string; file: File; preview: string; results: Detection[]; status: 'queued' | 'analyzing' | 'done' | 'error' | 'cancelled'; error?: string};
const supported = (f: File) => (/^image\/(jpeg|png|webp|avif)$/i.test(f.type) || /\.(png|jpe?g|jfif|webp|avif)$/i.test(f.name)) && f.size > 0 && f.size <= 20 * 1024 * 1024;
const number = (n: number | null) => n === null ? '—' : `${(n * 100).toFixed(1)}%`;
function download(data: unknown, name: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function App() {
  const [tab, setTab] = useState<Tab>('inspect');
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [settingsError, setSettingsError] = useState('');
  const [images, setImages] = useState<LabImage[]>([]);
  const [selectedId, setSelectedId] = useState('');
  const imagesRef = useRef<LabImage[]>([]);
  const previewUrls = useRef(new Set<string>());
  const selected = images.find(image => image.id === selectedId) ?? images[0];
  const file = selected?.file;
  const preview = selected?.preview ?? '';
  const results = selected?.results ?? [];
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState('');
  const [error, setError] = useState('');
  const [robustness, setRobustness] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [rows, setRows] = useState<BenchmarkRow[]>([]);
  const [failures, setFailures] = useState<string[]>([]);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const runId = useRef(0);
  const input = useRef<HTMLInputElement>(null);
  const folder = useRef<HTMLInputElement>(null);
  const variants: Variant[] = robustness ? ['original', 'jpeg75', 'resize512', 'reddit-webp640', 'reddit-webp960'] : ['original'];
  const activeModel = modelConfig(settings.model);
  useEffect(() => {
    if (isExtension) void getSettings().then(setSettings).catch(e => setSettingsError(String(e)));
    else {try {
      const stored = JSON.parse(localStorage.getItem('lens-settings:v2') ?? '{}');
      const current = stored.settingsVersion === 4;
      const model = current && Object.hasOwn(MODELS,stored.model) ? stored.model as ModelId : DEFAULT_SETTINGS.model;
      const threshold = current && stored.model === model && Number.isFinite(stored.threshold) && stored.threshold >= 0 && stored.threshold <= 1
        ? stored.threshold : modelConfig(model).threshold;
      const next:Settings={...DEFAULT_SETTINGS, model, threshold, imageSource: !current ? DEFAULT_SETTINGS.imageSource : stored.imageSource === 'displayed' ? 'displayed' : stored.imageSource === 'prefer-original' ? 'prefer-original' : isRedditStudent(model)?'displayed':'preview-first'};
      setSettings(next);localStorage.setItem('lens-settings:v2',JSON.stringify(next));
    } catch {}}
  }, []);
  useEffect(()=>{
    if(!isExtension)return;
    const changed=(changes:{[key:string]:chrome.storage.StorageChange},area:string)=>{
      if(area==='local'&&changes.settings)void getSettings().then(setSettings).catch(e=>setSettingsError(String(e)));
    };
    chrome.storage.onChanged.addListener(changed);return()=>chrome.storage.onChanged.removeListener(changed);
  },[]);
  useEffect(() => () => {
    runId.current++;
    resetAnalyzer();
    for (const url of previewUrls.current) URL.revokeObjectURL(url);
    previewUrls.current.clear();
  }, []);
  // Keep the listener current without reading or requesting clipboard permissions.
  useEffect(() => {
    function pasted(event: ClipboardEvent) {
      if (tab !== 'inspect' || event.defaultPrevented) return;
      const target = event.target;
      if (target instanceof Element && target.closest('input, textarea, [contenteditable="true"]')) return;
      const files = Array.from(event.clipboardData?.items ?? [])
        .filter(item => item.kind === 'file' && item.type.startsWith('image/'))
        .map(item => item.getAsFile()).filter((item): item is File => item !== null);
      if (!files.length) return;
      event.preventDefault(); chooseFiles(files);
    }
    window.addEventListener('paste', pasted);
    return () => window.removeEventListener('paste', pasted);
  });
  function replaceImages(next: LabImage[]) {imagesRef.current = next; setImages(next);}
  function updateImage(id: string, patch: Partial<LabImage>) {
    replaceImages(imagesRef.current.map(image => image.id === id ? {...image, ...patch} : image));
  }
  async function saveSettings(patch: Partial<Settings>) {
    const next = {...(isExtension?await getSettings():settings), ...patch}; setSettings(next);
    try {
      if (isExtension) await chrome.storage.local.set({settings: next});
      else localStorage.setItem('lens-settings:v2', JSON.stringify(next));
      setSettingsError('');
    } catch {setSettingsError('Settings could not be saved.');}
  }
  async function selectModel(model: ModelId) {
    runId.current++; resetAnalyzer('Detector changed.'); setBusy(false); setProgress('');
    replaceImages(imagesRef.current.map(image => ({...image, results: [], status: 'queued', error: undefined})));
    setRows([]); setFailures([]); setError('');
    await saveSettings({model, threshold: modelConfig(model).threshold,imageSource:sourceForModelSwitch(settings.imageSource,model)});
    if (isExtension) await chrome.storage.session.remove('results');
  }
  function chooseFiles(values: File[]) {
    if (!values.length) return;
    const eligible = values.filter(supported);
    const skipped = values.length - eligible.length;
    if (!eligible.length) {setError('Choose JPEG, PNG, WebP or AVIF images up to 20 MB each.'); return;}
    cancel();
    const added: LabImage[] = eligible.map(file => {
      const preview = URL.createObjectURL(file); previewUrls.current.add(preview);
      return {id: crypto.randomUUID(), file, preview, results: [], status: 'queued'};
    });
    const next = [...imagesRef.current, ...added]; replaceImages(next); setSelectedId(added[0].id);
    setError(skipped ? `${skipped} unsupported or oversized ${skipped === 1 ? 'file was' : 'files were'} skipped. Use JPEG, PNG, WebP or AVIF up to 20 MB each.` : '');
    void inspect(next.filter(image => image.status !== 'done' && image.status !== 'error'));
  }
  function removeImage(id: string) {
    cancel();
    const removed = imagesRef.current.find(image => image.id === id);
    if (removed) {URL.revokeObjectURL(removed.preview); previewUrls.current.delete(removed.preview);}
    replaceImages(imagesRef.current.filter(image => image.id !== id)); setError('');
  }
  function clearImages() {
    cancel();
    for (const url of previewUrls.current) URL.revokeObjectURL(url);
    previewUrls.current.clear(); replaceImages([]); setSelectedId(''); setError('');
  }
  async function diagnostic() {
    const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 480;
    const ctx = canvas.getContext('2d')!; ctx.fillStyle = '#e6dfc9'; ctx.fillRect(0, 0, 640, 480);
    ctx.fillStyle = '#bcc5ac'; ctx.fillRect(0, 320, 640, 160);
    ctx.fillStyle = '#a24b28'; ctx.beginPath(); ctx.arc(330, 210, 75, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = '#344d42'; ctx.fillRect(180, 280, 280, 18);
    ctx.fillStyle = '#fff8ec'; ctx.font = '18px "Google Sans"'; ctx.fillText('Diagnostic drawing · made with code', 170, 390);
    const blob = await new Promise<Blob>(resolve => canvas.toBlob(b => resolve(b!), 'image/png'));
    chooseFiles([new File([blob], 'diagnostic-drawing.png', {type: 'image/png'})]);
  }
  async function inspect(queue = imagesRef.current) {
    if (!queue.length) return;
    const id = ++runId.current; setBusy(true);
    const queuedIds = new Set(queue.map(image => image.id));
    replaceImages(imagesRef.current.map(image => queuedIds.has(image.id) ? {...image, status: 'queued', results: [], error: undefined} : image));
    try {
      for (let index = 0; index < queue.length; index++) {
        const image = queue[index]; if (id !== runId.current) return;
        updateImage(image.id, {status: 'analyzing'});
        const collected: Detection[] = [];
        try {
          for (const variant of variants) {
            setProgress(`${index + 1} / ${queue.length} · ${image.file.name}${variant === 'original' ? '' : ` · ${variantLabel(variant)}`}`);
            const result = await analyze(image.file, variant, settings.model); if (id !== runId.current) return;
            collected.push(result); updateImage(image.id, {results: [...collected]});
          }
          updateImage(image.id, {status: 'done'});
        } catch (e) {
          if (id !== runId.current) return;
          updateImage(image.id, {status: 'error', error: e instanceof Error ? e.message : String(e)});
        }
      }
    } finally {if (id === runId.current) {setBusy(false); setProgress('');}}
  }
  function cancel() {
    runId.current++; resetAnalyzer(); setBusy(false); setProgress('');
    replaceImages(imagesRef.current.map(image => image.status === 'queued' || image.status === 'analyzing' ? {...image, status: 'cancelled'} : image));
  }
  async function benchmark() {
    const id = ++runId.current; setBusy(true); setError(''); setRows([]); setFailures([]);
    const output: BenchmarkRow[] = [], errors: string[] = [];
    const hashes = new Map<string, string>();
    try {
      for (let index = 0; index < batchFiles.length; index++) {
        const item = batchFiles[index];
        const parts = item.webkitRelativePath.split('/');
        const splitIndex = parts.findIndex(p => p === 'calibration' || p === 'validation');
        const split = parts[splitIndex];
        const labelIndex = splitIndex + 1;
        if (splitIndex < 0 || !['real', 'ai'].includes(parts[labelIndex])) {errors.push(`${item.name}: missing split or label folder`); continue;}
        setProgress(`${index + 1} / ${batchFiles.length} · ${item.name}`);
        try {
          const hashBuffer = await crypto.subtle.digest('SHA-256', await item.arrayBuffer());
          const sourceHash = Array.from(new Uint8Array(hashBuffer), b => b.toString(16).padStart(2, '0')).join('');
          if (hashes.has(sourceHash)) throw new Error(`duplicate source image (already seen in ${hashes.get(sourceHash)})`);
          hashes.set(sourceHash, item.webkitRelativePath);
          const itemRows: BenchmarkRow[] = [];
          for (const variant of variants) {
            const result = await analyze(item, variant, settings.model); if (id !== runId.current) return;
            itemRows.push({...result, file: item.webkitRelativePath, label: parts[labelIndex] as 'real' | 'ai', group: parts.length > labelIndex + 2 ? parts[labelIndex + 1] : 'unspecified', split: split as 'calibration' | 'validation', sourceHash});
          }
          output.push(...itemRows); setRows([...output]);
        } catch (e) {if (id !== runId.current) return; errors.push(`${item.name}: ${e instanceof Error ? e.message : String(e)}`);}
        setFailures([...errors]);
      }
    } finally {if (id === runId.current) {setBusy(false); setProgress(''); setFailures(errors);}}
  }
  function dropped(e: DragEvent) {e.preventDefault(); setDragging(false); chooseFiles(Array.from(e.dataTransfer.files));}
  const primary = results.find(r => r.variant === 'original');
  const unstable = primary && results.some(r => (r.score >= settings.threshold) !== (primary.score >= settings.threshold));
  const originalRows = rows.filter(r => r.variant === 'original');
  const validationRows = originalRows.filter(r => r.split === 'validation');
  const summary = metrics(validationRows, settings.threshold);
  const exportScores = () => download({schemaVersion: 1, model: activeModel, operatingPoint: settings.threshold, rows, failures, exportedAt: new Date().toISOString()}, 'reddit-lens-scores.json');
  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e => {e.preventDefault(); setTab('inspect');}} aria-label="Reddit Lens home"><span className="brand-mark"><ScanLine size={24}/></span><span>Reddit Lens</span></a>
      <div className="sidebar-caption">Workspace</div>
      <nav aria-label="Workspace">{([
        ['inspect', ScanLine, 'Image lab'], ['benchmark', FlaskConical, 'Benchmarks'], ['settings', SlidersHorizontal, 'Settings'],
      ] as const).map(([key, Icon, label]) => <button key={key} className={tab === key ? 'nav-item active' : 'nav-item'} aria-current={tab === key ? 'page' : undefined} onClick={() => setTab(key)}><Icon size={18}/>{label}{tab === key && <ChevronRight size={15}/>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="local-tag"><span/> Runs on your device</div><p>Images stay on your device.</p><a href={activeModel.repository} target="_blank" rel="noreferrer">Model & source <ArrowUpRight size={13}/></a></div>
    </aside>
    <main>
      <div className="topbar"><span>Workspace <ChevronRight size={13}/> {tab === 'inspect' ? 'Image lab' : tab === 'benchmark' ? 'Benchmarks' : 'Settings'}</span><span className="preview-label">Experimental</span></div>
      <div className="workspace">
        <header className="page-header"><div><h1>{tab === 'inspect' ? 'Image lab' : tab === 'benchmark' ? 'Benchmarks' : 'Settings'}</h1><p>{tab === 'inspect' ? 'Check images for AI signals. Upload, drop or paste to begin.' : tab === 'benchmark' ? 'Measure detections and false flags on images with known origins.' : 'Manage your detector and Reddit scanning preferences.'}</p></div>{tab === 'inspect' && <label className="lab-model-select">Detection model<select aria-label="Detection model" value={settings.model} onChange={e => void selectModel(e.target.value as ModelId)}>{Object.values(MODELS).map(model => <option key={model.id} value={model.id}>{model.shortName}</option>)}</select></label>}</header>
        {tab === 'inspect' && <>
          <div className="lab-layout">
            <section className="image-workspace" aria-label="Image input">
              <div className="section-heading"><span>Your images</span>{images.length ? <button className="text-button" onClick={() => input.current?.click()}><ImagePlus size={14}/> Add images</button> : <span className="subtle">JPEG · PNG · WebP · AVIF</span>}</div>
              <input ref={input} type="file" hidden multiple aria-label="Choose images" accept="image/jpeg,image/png,image/webp,image/avif,.jfif" onChange={e => {chooseFiles(Array.from(e.target.files ?? [])); e.target.value = '';}}/>
              <div className={`dropzone ${preview ? 'has-image' : ''} ${dragging ? 'dragging' : ''}`} onDragOver={e => {e.preventDefault(); setDragging(true);}} onDragLeave={() => setDragging(false)} onDrop={dropped}>
                {preview ? <><img src={preview} alt={`Selected image: ${file?.name}`}/><button className="remove-image" aria-label="Remove image" onClick={() => selected && removeImage(selected.id)}><X size={17}/></button><div className="image-caption"><span>{file?.name}</span><span>{((file?.size ?? 0) / 1024).toFixed(0)} KB</span></div></> : <div className="empty-image"><div className="upload-mark"><ImagePlus size={28} strokeWidth={1.5}/></div><h2>Drop images here</h2><p>or paste from your clipboard with ⌘V / Ctrl+V</p><button className="button primary" onClick={() => input.current?.click()}>Choose images <ArrowRight size={16}/></button><span className="upload-note">Up to 20 MB each · at least 128 × 128 px</span></div>}
              </div>
              <div className="image-controls"><details className="analysis-options"><summary>Analysis options</summary><label className="checkbox-label"><input type="checkbox" checked={robustness} disabled={busy} onChange={e => setRobustness(e.target.checked)}/><span>Compare compressed &amp; resized copies<small>Original, JPEG, 512 px and WebP 640 / 960 px</small></span></label><button className="text-button" onClick={() => void diagnostic()}>Try a diagnostic drawing <ArrowUpRight size={12}/></button></details>{file && <button className="button secondary" disabled={busy} onClick={() => void inspect()}>{busy ? <LoaderCircle className="spin" size={16}/> : <ScanLine size={16}/>} {busy ? 'Analyzing' : images.length > 1 ? 'Analyze all again' : 'Analyze again'}</button>}</div>
              {busy && <div className="progress" role="status"><LoaderCircle className="spin" size={14}/>{progress}<button onClick={cancel}>Cancel</button></div>}
              {error && <div role="alert" className="error-message">{error}</div>}
              {selected?.error && <div role="alert" className="error-message">{selected.file.name}: {selected.error}</div>}
            </section>
            <aside className="inspector" aria-label="Analysis result"><div className="section-heading"><span>Analysis</span></div>
              <div className={`result ${primary ? 'ready' : ''}`} data-band={primary ? decisionBand(primary.score, settings.threshold, activeModel.uncertainThreshold, primary.semanticScore) : 'empty'}>
                <div className="result-symbol"><ScanLine size={25} strokeWidth={1.2}/></div>
                <h2>{primary ? verdict(primary.score, settings.threshold, activeModel.uncertainThreshold, primary.semanticScore) : selected?.status === 'analyzing' ? 'Analyzing image…' : selected?.status === 'queued' ? 'Image queued' : selected?.status === 'error' ? 'Analysis failed' : selected?.status === 'cancelled' ? 'Analysis stopped' : 'Awaiting an image'}</h2>
                <p>{primary ? 'A model signal, not a verified origin. An unflagged image can still be AI-generated.' : 'Choose an image to see its result.'}</p>
                <div className="score-line"><span>AI signal score</span><strong>{primary ? `${(primary.score * 100).toFixed(1)}%` : '—'}</strong></div>
                <div className="signal-track"><div style={{width: primary ? `${primary.score * 100}%` : '0%'}}/><span style={{left: `${settings.threshold * 100}%`}}/></div><div className="scale-labels"><span>0%</span><span>Cutoff {(settings.threshold * 100).toFixed(1)}%</span><span>100%</span></div>
                {primary && <><div className="analysis-meta"><span>{primary.width} × {primary.height}</span><span>{primary.engine.toUpperCase()} · {(primary.elapsedMs / 1000).toFixed(2)} s</span></div><p className="score-warning">A model score, not a probability of AI origin.</p><details className="signal-details"><summary>How this score is read</summary><p className="subtle">Raw score {primary.score.toFixed(6)} · cutoff {settings.threshold.toFixed(3)}</p>{activeModel.uncertainThreshold !== undefined && <p className="subtle">Semantic score at or above {Math.min(activeModel.uncertainThreshold,settings.threshold).toFixed(3)}: uncertain. Combined score at or above the operating point: strong AI signal. A lower score does not verify an image’s origin.</p>}{settings.model === 'community-ferret-preview-v1' && primary.components && <div className="component-signals">{(['community_forensics','ferretnet_b'] as const).map(key => <div key={key}><span>{key === 'community_forensics' ? 'Community Preview' : 'FerretNet'}<small>Threshold {key === 'community_forensics' ? '0.500' : '0.830'}</small></span><strong>{primary.components![key].score.toFixed(4)}</strong></div>)}</div>}{settings.model === 'ferretnet-opensynthid-v1' && primary.components && <div className="component-signals">{(['ferretnet_b', 'opensynthid'] as const).map(key => <div key={key}><span>{key === 'ferretnet_b' ? 'FerretNet' : 'OpenSynthID'}<small>Threshold {Number(primary.components![key].threshold).toFixed(3)}</small></span><strong>{primary.components![key].score.toFixed(4)}</strong></div>)}<p className="subtle">The combined signal uses either detector. OpenSynthID does not verify image provenance.</p></div>}</details></>}
              </div>
              <div className="engine-info"><span className="eyebrow">Current model</span><h3>{activeModel.shortName}</h3><button className="text-button" onClick={() => setTab('settings')}>Model settings <ArrowRight size={13}/></button></div>
            </aside>
          </div>
          {images.length > 0 && <section className="image-gallery" aria-label="Image results">
            <div className="gallery-heading"><div><h2>All images</h2><p role="status">{images.filter(image => image.status === 'done').length} of {images.length} analyzed · {images.filter(image => image.results.some(result => result.variant === 'original' && result.score >= settings.threshold)).length} flagged</p></div><div className="gallery-actions"><button className="text-button" disabled={busy || !images.some(image => image.results.length)} onClick={() => download({model: activeModel, threshold: settings.threshold, images: images.map(({file, results, status, error}) => ({file: file.name, results, status, error}))}, 'reddit-lens-images.json')}><ArrowDownToLine size={14}/> Export batch</button><button className="text-button" onClick={clearImages}><X size={14}/> Clear all</button></div></div>
            <div className="image-grid">{images.map(image => {
              const result = image.results.find(result => result.variant === 'original');
              const flagged = result && result.score >= settings.threshold;
              const uncertain = result && decisionBand(result.score, settings.threshold, activeModel.uncertainThreshold, result.semanticScore) === 'uncertain';
              const label = result ? flagged ? 'AI signal detected' : uncertain ? 'Uncertain AI signal' : 'No strong AI signal' : image.status === 'analyzing' ? 'Analyzing…' : image.status === 'error' ? 'Analysis failed' : image.status === 'cancelled' ? 'Analysis stopped' : 'Queued';
              return <button key={image.id} className={`image-tile ${selected?.id === image.id ? 'selected' : ''} ${flagged ? 'flagged' : uncertain ? 'uncertain' : ''}`} aria-pressed={selected?.id === image.id} aria-label={`${image.file.name}: ${label}${result ? `, AI score ${(result.score * 100).toFixed(1)}%` : ''}`} onClick={() => setSelectedId(image.id)}>
                <div className="tile-preview"><img src={image.preview} alt="" loading="lazy"/>{result ? <span className="tile-score"><strong>{(result.score * 100).toFixed(1)}%</strong><small>AI score</small></span> : <span className="tile-score pending">{image.status === 'queued' || image.status === 'analyzing' ? <LoaderCircle size={16} className={image.status === 'analyzing' ? 'spin' : ''}/> : <X size={16}/>}</span>}</div>
                <div className="tile-details"><span className="tile-verdict"><span className={`verdict-dot ${result ? flagged ? 'ai' : uncertain ? 'uncertain' : 'clear' : ''}`}/>{label}</span><span className="tile-name" title={image.file.name}>{image.file.name}</span>{image.error ? <small className="tile-error">{image.error}</small> : result && <small>{result.width} × {result.height} · {(result.elapsedMs / 1000).toFixed(2)} s{image.status === 'cancelled' ? ' · Checks stopped' : image.status === 'analyzing' ? ' · Checking variants' : ''}</small>}</div>
              </button>;
            })}</div>
            <p className="subtle gallery-note">Percentages show the raw model score, not the probability of AI origin. Select an image for its detailed analysis.</p>
          </section>}
          {results.length > 0 && <details className="robustness-results"><summary>Score details &amp; export</summary><div className="section-heading"><span><Layers3 size={17}/> {results.length > 1 ? 'RESILIENCE CHECK' : 'IMAGE SCORE'}</span><button className="text-button" onClick={() => download({model: activeModel, results, threshold: settings.threshold}, 'reddit-lens-image.json')}><ArrowDownToLine size={14}/> Export result</button></div><div className="variant-grid">{results.map(result => <div key={result.variant}><span>{variantLabel(result.variant)}</span><strong>{result.score.toFixed(4)}</strong><div className="mini-track"><span style={{width: `${result.score * 100}%`}}/></div><small>{verdict(result.score, settings.threshold, activeModel.uncertainThreshold, result.semanticScore)}</small></div>)}</div>{unstable && <p className="instability">The flag changes after transformation. Treat this result as unstable.</p>}<p className="subtle">Preview checks simulate width-limited WebP at quality 80; they do not reproduce every Reddit codec setting. Repeat checks by the same model do not establish accuracy.</p></details>}
          <details className="field-notes"><summary>About this detector</summary><div><p>{settings.model === 'reddit-sherlock-2' ? 'Sherlock 2 is the stricter follow-up. On new held-out Reddit images it flagged about 0.1% of real images, compared with about 0.5% for Sherlock 1.0, and caught about 58% of AI images compared with about 66%. Art still causes most of its false flags. Use it for experimental comparison.' : settings.model === 'reddit-sherlock-1.0' ? 'Sherlock 1.0 is a fully fine-tuned Reddit preview student. On held-out Reddit images it caught about 65% of AI images while flagging about 0.5% of real ones, but those labels come from subreddit sources and real browsing is harder. Art and photos cause most false flags. Use it for experimental comparison.' : settings.model === 'reddit-student-last2-v1' ? 'The Reddit student uses a conservative cutoff selected on source-labelled Reddit previews. Many AI images, especially difficult photoreal images, are still missed. Original-file recall can be lower than Community + Ferret. Use it for experimental comparison.' : settings.model.startsWith('community-') ? 'Community Preview uses a frozen small backbone and a head trained on development images with identical JPEG / WebP processing for both classes. Its higher and uncertain cutoffs are experimental; source bias and missed AI images remain possible.' : settings.model.startsWith('mirror-') ? 'MIRROR is a large experimental local detector. It sees a 224 px center crop and can miss edits outside that region. Its score is not a probability; the combination can increase both detection and false flags.' : settings.model === 'ferretnet-opensynthid-v1' ? 'FerretNet checks a native center crop. OpenSynthID adds a resized image and frequency analysis for additional Gemini signals. Both can miss AI images and falsely flag authentic artwork.' : 'Compression, artwork and new generators can confuse the ensemble. It checks resized features and native-resolution regions, and small edits may go undetected.'} Benchmark on representative images before relying on flags.</p></div><button onClick={() => setTab('benchmark')} className="text-button">Open benchmarks <ArrowRight size={15}/></button></details>
        </>}
        {tab === 'benchmark' && <section className="benchmark-workspace">
          <ValidationBenchmark activeModel={settings.model}/>
          <div className="benchmark-intro"><div><span className="eyebrow">Labeled images</span><h2>Choose a labeled collection</h2><p>Keep calibration and validation separate. Each image needs a known origin, not a detector’s guess.</p><code>dataset/calibration/real/photos/image.jpg<br/>dataset/calibration/ai/flux/image.jpg<br/>dataset/validation/real/artwork/image.jpg<br/>dataset/validation/ai/gpt-image/image.jpg</code></div><div className="folder-actions"><input ref={folder} type="file" hidden multiple {...({webkitdirectory: ''} as object)} onChange={e => {setBatchFiles(Array.from(e.target.files ?? []).filter(supported)); setRows([]); setFailures([]); setError('');}}/><button className="button secondary" disabled={busy} onClick={() => folder.current?.click()}>Choose dataset folder <ArrowUpRight size={15}/></button><span>{batchFiles.length ? `${batchFiles.length} eligible images selected` : 'No images leave your device'}</span><label className="checkbox-label"><input type="checkbox" checked={robustness} disabled={busy} onChange={e => setRobustness(e.target.checked)}/>Include resilience checks</label><button className="button primary" disabled={!batchFiles.length || busy} onClick={() => void benchmark()}>{busy ? <LoaderCircle className="spin" size={15}/> : <FlaskConical size={15}/>} Run benchmark</button></div></div>
          {busy && <div className="progress" role="status"><LoaderCircle className="spin" size={14}/>{progress}<button onClick={cancel}>Cancel</button></div>}
          <div className="benchmark-stats">{[['Validation images', validationRows.length], ['False-positive rate', number(summary.fpr)], ['AI recall', number(summary.recall)], ['Flag precision', number(summary.precision)]].map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
          <div className="section-heading"><span>Image results</span><button className="text-button" disabled={!rows.length || busy} onClick={exportScores}><ArrowDownToLine size={14}/> Export scores</button></div>
          {rows.length ? <div className="table-scroll"><table><thead><tr><th>Image</th><th>Origin / split</th><th>Variant</th><th>Score</th><th>Flag</th></tr></thead><tbody>{rows.map(row => <tr key={`${row.file}:${row.variant}`}><td title={row.file}>{row.file.split('/').at(-1)}</td><td>{row.label} / {row.split}</td><td>{row.variant}</td><td className="mono">{row.score.toFixed(4)}</td><td>{row.score >= settings.threshold ? 'AI signal' : 'No strong signal'}</td></tr>)}</tbody></table></div> : <div className="empty-benchmark"><FlaskConical size={26} strokeWidth={1.2}/><p>No benchmark results yet.</p><span>Run a collection to see scores, false flags and missed detections.</span></div>}
          {failures.length > 0 && <div className="error-message" role="alert"><strong>{failures.length} images skipped or failed</strong><ul>{failures.map((failure, i) => <li key={i}>{failure}</li>)}</ul><p>Resolve these errors before treating this collection as a validation benchmark.</p></div>}
          <div className="field-notes"><span className="notes-number">i</span><div><h3>A small sample can’t establish reliability.</h3><p>Summary metrics use original validation images at the current operating point. Export scores, then run <code>npm run evaluate -- scores.json</code> to choose a threshold on calibration images and assess held-out validation, including each category and transformation. Results are not applied automatically.</p></div></div>
        </section>}
        {tab === 'settings' && <section className="settings-workspace">
          <div className="settings-section"><div><Layers3 size={21}/><h2>Detection model</h2><p>Choose the detector used in the lab and on Reddit. Models load locally; larger models take longer to start.</p></div><div className="settings-fields model-choice" role="group" aria-label="Detection model">{Object.values(MODELS).map(model => <button key={model.id} className={settings.model === model.id ? 'model-option active' : 'model-option'} aria-pressed={settings.model === model.id} onClick={() => void selectModel(model.id)}><strong>{model.name}</strong><small>{model.description}</small></button>)}</div></div>
          <div className="settings-section"><div><Settings2 size={21}/><h2>Reddit scanning</h2><p>The Reddit status panel shows when scanning is running. Each scanned image gets a badge, including low-score images.</p>{!isExtension && <span className="inline-note">Available after loading the Chrome extension.</span>}</div><div className="settings-fields"><label className="toggle-row"><span>Run Reddit Lens<small>Enable detection on visible images in opened posts</small></span><input type="checkbox" role="switch" disabled={!isExtension} checked={settings.enabled} onChange={e => void saveSettings({enabled: e.target.checked})}/></label><label className="toggle-row"><span>Scan the Reddit feed<small>{settings.enabled?'Also scan images as you scroll the feed':'Turn on Run Reddit Lens first'}</small></span><input type="checkbox" role="switch" disabled={!isExtension||!settings.enabled} checked={settings.feed} onChange={e => void saveSettings({feed: e.target.checked})}/></label><ScanSettings settings={settings} save={saveSettings} extension={isExtension}/></div></div>
          <div className="settings-section"><div><SlidersHorizontal size={21}/><h2>Detection cutoff</h2><p>A higher raw-score threshold yields fewer flags. It does not guarantee fewer mistakes on new images.</p></div><div className="settings-fields"><label className="threshold-label" htmlFor="threshold">Flag images at or above <strong>{settings.threshold.toFixed(3)}</strong></label><input id="threshold" type="range" min="0" max="1" step="0.001" value={settings.threshold} onChange={e => void saveSettings({threshold: Number(e.target.value)})}/><p className="subtle">The default {activeModel.threshold.toFixed(3)} is provisional. It is an operating point, not a confidence level.</p><button className="text-button" onClick={() => void saveSettings({threshold: activeModel.threshold})}>Reset operating point</button></div></div>
          <div className="settings-section"><div><ShieldCheck size={21}/><h2>Model & privacy</h2><p>The selected model runs in your local backend. Images stay on your device.</p></div><div className="settings-fields model-details"><span>Model<strong>{activeModel.name}</strong></span><span>Runtime<strong>Local Python · MPS / CPU</strong></span><span>License<strong>{activeModel.license}</strong></span><span>Scope<strong>Still images · Reddit hosts</strong></span><a href={activeModel.repository} target="_blank" rel="noreferrer">Read the model source <ArrowUpRight size={14}/></a></div></div>
          {settingsError && <p role="alert" className="error-message">{settingsError}</p>}
          {isExtension && <button className="button secondary" onClick={() => void chrome.storage.session.remove('results').then(() => {setSettingsError('Session score cache cleared.');})}>Clear Reddit score cache</button>}
        </section>}
        <footer className="workspace-footer"><span><ShieldCheck size={14}/> Local analysis</span><span>Scores are signals. Origin needs evidence.</span></footer>
      </div>
    </main>
  </div>;
}
export default App;
