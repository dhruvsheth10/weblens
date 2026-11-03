import React, {createContext, useContext} from 'react';
import {AbsoluteFill, interpolate, OffthreadVideo, Sequence, staticFile, useCurrentFrame} from 'remotion';
import {color, FONT, FPS, rise, sec} from './theme';

const SceneStart = createContext(0);

/** Global timeline seconds, so scenes can key animations to narration timestamps. */
export const useTime = () => useCurrentFrame() / FPS + useContext(SceneStart);

const EDGE = 0.3;

export const Scene: React.FC<{start: number; end: number; children: React.ReactNode}> = ({start, end, children}) => {
  const from = start - EDGE;
  return (
    <Sequence from={sec(from)} durationInFrames={sec(end - start + EDGE * 2)}>
      <SceneStart.Provider value={from}>
        <SceneFade start={start} end={end}>{children}</SceneFade>
      </SceneStart.Provider>
    </Sequence>
  );
};

const SceneFade: React.FC<{start: number; end: number; children: React.ReactNode}> = ({start, end, children}) => {
  const t = useTime();
  const opacity = interpolate(t, [start - EDGE, start + EDGE, end - EDGE, end + EDGE], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const scale = interpolate(t, [start - EDGE, start + EDGE * 2], [1.03, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const blur = interpolate(t, [start - EDGE, start + EDGE, end - EDGE, end + EDGE], [8, 0, 0, 8], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return <AbsoluteFill style={{opacity, transform: `scale(${scale})`, filter: blur > 0.05 ? `blur(${blur}px)` : undefined}}>{children}</AbsoluteFill>;
};

export const Stage: React.FC = () => {
  const t = useCurrentFrame() / FPS;
  const drift = Math.sin(t / 9) * 60;
  return (
    <AbsoluteFill style={{background: color.stage}}>
      <AbsoluteFill style={{background: `radial-gradient(900px 700px at ${18 + drift / 40}% ${12 + drift / 60}%, rgba(101,80,173,0.32), transparent 70%)`}}/>
      <AbsoluteFill style={{background: `radial-gradient(800px 640px at ${88 - drift / 50}% ${92}%, rgba(20,119,111,0.20), transparent 70%)`}}/>
      <AbsoluteFill style={{backgroundImage: 'radial-gradient(rgba(255,255,255,0.045) 1px, transparent 1px)', backgroundSize: '28px 28px'}}/>
    </AbsoluteFill>
  );
};

export const Window: React.FC<{url: string; width: number; height: number; children: React.ReactNode; style?: React.CSSProperties}> = ({url, width, height, children, style}) => (
  <div style={{width, height: height + 44, borderRadius: 18, overflow: 'hidden', background: '#fff', boxShadow: '0 40px 120px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.08)', ...style}}>
    <div style={{height: 44, background: '#ecebf2', display: 'flex', alignItems: 'center', padding: '0 18px', gap: 8}}>
      {['#ff5f57', '#febc2e', '#28c840'].map((dot) => <div key={dot} style={{width: 12, height: 12, borderRadius: 6, background: dot}}/>)}
      <div style={{marginLeft: 18, flex: 1, maxWidth: 520, height: 26, borderRadius: 13, background: '#fff', color: '#656579', fontFamily: FONT, fontSize: 14, display: 'flex', alignItems: 'center', padding: '0 14px'}}>{url}</div>
    </div>
    <div style={{width, height, position: 'relative', overflow: 'hidden'}}>{children}</div>
  </div>
);

/** A cropped screen recording. `zoom` scales around (`x`, `y`) given as fractions of the frame. */
export const Clip: React.FC<{name: string; rate?: number; trim?: number; zoom?: number; x?: number; y?: number}> = ({name, rate = 1, trim = 0, zoom = 1, x = 0.5, y = 0.5}) => (
  <AbsoluteFill style={{transform: `scale(${zoom})`, transformOrigin: `${x * 100}% ${y * 100}%`}}>
    <OffthreadVideo src={staticFile(`clips/${name}.mp4`)} playbackRate={rate} trimBefore={sec(trim)} muted style={{width: '100%', height: '100%', objectFit: 'cover'}}/>
  </AbsoluteFill>
);

export const Kicker: React.FC<{children: React.ReactNode; style?: React.CSSProperties}> = ({children, style}) => (
  <div style={{fontFamily: FONT, fontSize: 26, fontWeight: 600, letterSpacing: 4, textTransform: 'uppercase', color: color.lavender, ...style}}>{children}</div>
);

export const Headline: React.FC<{children: React.ReactNode; size?: number; style?: React.CSSProperties}> = ({children, size = 84, style}) => (
  <div style={{fontFamily: FONT, fontSize: size, fontWeight: 600, letterSpacing: -size * 0.025, lineHeight: 1.08, color: color.text, ...style}}>{children}</div>
);

/** Fades and lifts its children in at `at` seconds on the global timeline. */
export const Reveal: React.FC<{at: number; children: React.ReactNode; y?: number; style?: React.CSSProperties}> = ({at, children, y = 28, style}) => {
  const p = rise(useTime(), at, 0.7);
  return <div style={{opacity: p, transform: `translateY(${(1 - p) * y}px)`, filter: p < 0.98 ? `blur(${(1 - p) * 6}px)` : undefined, ...style}}>{children}</div>;
};

export const Chip: React.FC<{children: React.ReactNode; tone?: 'teal' | 'purple' | 'plain'; style?: React.CSSProperties}> = ({children, tone = 'plain', style}) => {
  const tones = {
    teal: {background: 'rgba(95,211,192,0.12)', border: 'rgba(95,211,192,0.45)', dot: color.tealBright},
    purple: {background: 'rgba(185,168,255,0.12)', border: 'rgba(185,168,255,0.45)', dot: color.lavender},
    plain: {background: 'rgba(255,255,255,0.06)', border: 'rgba(255,255,255,0.16)', dot: '#ffffff'},
  }[tone];
  return (
    <div style={{display: 'inline-flex', alignItems: 'center', gap: 14, padding: '14px 26px', borderRadius: 999, background: tones.background, border: `1.5px solid ${tones.border}`, fontFamily: FONT, fontSize: 30, fontWeight: 500, color: color.text, ...style}}>
      <div style={{width: 10, height: 10, borderRadius: 5, background: tones.dot}}/>
      {children}
    </div>
  );
};

const CORNERS = ['M10 18v-5a3 3 0 0 1 3-3h5', 'M30 10h5a3 3 0 0 1 3 3v5', 'M38 30v5a3 3 0 0 1-3 3h-5', 'M18 38h-5a3 3 0 0 1-3-3v-5'];

export const LensMark: React.FC<{size: number; progress?: number}> = ({size, progress = 1}) => (
  <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
    <rect width="48" height="48" rx="12" fill="rgba(185,168,255,0.14)"/>
    <g transform={`translate(24 24) scale(${0.6 + 0.4 * progress}) translate(-24 -24)`} stroke={color.lavender} strokeWidth="2.6" strokeLinecap="round" opacity={progress}>
      {CORNERS.map((d) => <path key={d} d={d}/>)}
    </g>
    <line x1={24 - 7 * progress} y1="24" x2={24 + 7 * progress} y2="24" stroke={color.lavender} strokeWidth="2.6" strokeLinecap="round"/>
  </svg>
);

/** A highlight ring that draws around a region of a recording. Coordinates are in px of the parent. */
export const Ring: React.FC<{at: number; left: number; top: number; width: number; height: number; tone?: string}> = ({at, left, top, width, height, tone = color.lavender}) => {
  const p = rise(useTime(), at, 0.5);
  return <div style={{position: 'absolute', left: left - 8, top: top - 8, width: width + 16, height: height + 16, borderRadius: 14, border: `3px solid ${tone}`, opacity: p, transform: `scale(${1.15 - p * 0.15})`, boxShadow: `0 0 40px ${tone}88`}}/>;
};
