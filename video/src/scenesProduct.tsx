import React from 'react';
import {AbsoluteFill, Img, staticFile} from 'remotion';
import {Chip, Clip, Headline, Kicker, LensMark, Reveal, useTime, Window} from './components';
import {between, color, FONT, rise} from './theme';

const TILE_COLUMNS: string[][] = [
  ['t-street', 't-pinup', 't-crab'],
  ['t-lightning', 't-miko', 't-autumn'],
  ['t-gold', 't-throne', 't-drawing'],
  ['t-lion', 't-autumn', 't-street'],
  ['t-crab', 't-drawing', 't-lightning'],
];

export const Hook: React.FC = () => {
  const t = useTime();
  const lift = t * 34;
  const ring = rise(t, 4.3, 0.6);
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{perspective: 1800}}>
        <div style={{position: 'absolute', left: -120, top: -260, display: 'flex', gap: 28, transform: `rotateX(16deg) rotateZ(-7deg) translateY(${-lift}px)`, transformOrigin: '50% 0'}}>
          {TILE_COLUMNS.map((column, i) => (
            <div key={i} style={{display: 'flex', flexDirection: 'column', gap: 28, marginTop: (i % 2) * 140}}>
              {[...column, ...column].map((name, j) => {
                const focus = name === 't-throne' && j === 1;
                return (
                  <div key={j} style={{position: 'relative', width: 400, borderRadius: 18, overflow: 'visible'}}>
                    <Img src={staticFile(`stills/${name}.png`)} style={{width: 400, borderRadius: 18, display: 'block', opacity: focus ? 1 : 0.55 + 0.45 * (1 - ring), boxShadow: '0 20px 60px rgba(0,0,0,0.5)'}}/>
                    {focus ? (
                      <>
                        <div style={{position: 'absolute', inset: -10, borderRadius: 24, border: `4px solid ${color.lavender}`, opacity: ring, boxShadow: `0 0 60px ${color.lavender}`}}/>
                        <div style={{position: 'absolute', right: -24, top: -30, padding: '10px 22px', borderRadius: 999, background: color.lavender, color: color.ink, fontFamily: FONT, fontWeight: 700, fontSize: 30, opacity: ring, transform: `scale(${0.6 + 0.4 * ring})`}}>AI?</div>
                      </>
                    ) : null}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </AbsoluteFill>
      <AbsoluteFill style={{background: 'linear-gradient(90deg, rgba(18,17,25,0.92) 0%, rgba(18,17,25,0.55) 45%, rgba(18,17,25,0.1) 100%)'}}/>
      <AbsoluteFill style={{justifyContent: 'center', padding: '0 140px'}}>
        <Reveal at={1.6}><Kicker>Scroll for a minute</Kicker></Reveal>
        <Reveal at={6.0} style={{marginTop: 26}}><Headline size={96}>Most AI images come<br/>with no label at all.</Headline></Reveal>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

export const Intro: React.FC = () => {
  const t = useTime();
  const mark = rise(t, 9.6, 0.9);
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', textAlign: 'center'}}>
      <Reveal at={8.6}><Kicker>Hi, I'm Dhruv</Kicker></Reveal>
      <div style={{display: 'flex', alignItems: 'center', gap: 40, marginTop: 40, opacity: mark, transform: `scale(${0.9 + 0.1 * mark})`}}>
        <LensMark size={150} progress={mark}/>
        <Headline size={170}>WebLens</Headline>
      </div>
      <Reveal at={10.6} style={{marginTop: 36}}>
        <div style={{fontFamily: FONT, fontSize: 44, color: color.muted}}>A Chrome extension that checks images for signs of AI</div>
      </Reveal>
      <Reveal at={14.1} style={{marginTop: 56}}><Chip tone="teal">Runs right on your own computer</Chip></Reveal>
    </AbsoluteFill>
  );
};

export const Audience: React.FC = () => {
  const t = useTime();
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{transform: `scale(${1.08 + (t - 16) * 0.012})`}}>
        <Img src={staticFile('stills/t-lightning.png')} style={{width: '100%', height: '100%', objectFit: 'cover', filter: 'blur(3px) brightness(0.45)'}}/>
      </AbsoluteFill>
      <AbsoluteFill style={{background: 'radial-gradient(circle at 50% 50%, rgba(18,17,25,0.2), rgba(18,17,25,0.85))'}}/>
      <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', textAlign: 'center'}}>
        <Reveal at={16.3}><Kicker>For everyone online</Kicker></Reveal>
        <Reveal at={17.5} style={{marginTop: 30}}><Headline size={110}>A second look</Headline></Reveal>
        <Reveal at={18.4} style={{marginTop: 8}}><Headline size={110} style={{color: color.lavender}}>before you trust or share.</Headline></Reveal>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

const FeedWindow: React.FC<{children: React.ReactNode; label: string; at: number; tone: 'teal' | 'purple' | 'plain'}> = ({children, label, at, tone}) => (
  <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
    <Window url="reddit.com" width={1440} height={810}>{children}</Window>
    <Reveal at={at} style={{position: 'absolute', bottom: 34}}><Chip tone={tone}>{label}</Chip></Reveal>
  </AbsoluteFill>
);

// Label positions inside the 1920×1080 opened-post crops.
const POST_LABEL = {x: 0.22, y: 0.15};

export const FeedBrowse: React.FC = () => (
  <FeedWindow label="Browse like normal" at={20.7} tone="plain">
    <Clip name="feed-real" trim={0.3}/>
  </FeedWindow>
);

export const FeedClear: React.FC = () => {
  const t = useTime();
  return (
    <FeedWindow label="Every image gets a label and a score" at={22.6} tone="teal">
      <Clip name="post-clear" trim={0.4} zoom={between(t, 22.7, 23.9, 1, 1.75)} x={POST_LABEL.x} y={POST_LABEL.y}/>
    </FeedWindow>
  );
};

export const FeedAi: React.FC = () => (
  <FeedWindow label="Strong AI signal" at={25.3} tone="purple">
    <Clip name="feed-ai" trim={0.2} rate={1.2}/>
  </FeedWindow>
);

export const FeedFlag: React.FC = () => {
  const t = useTime();
  return (
    <FeedWindow label="WebLens flags it" at={27.9} tone="purple">
      <Clip name="post-flagged" trim={0.3} zoom={between(t, 27.7, 28.9, 1, 1.75)} x={POST_LABEL.x} y={POST_LABEL.y}/>
    </FeedWindow>
  );
};

export const Lab: React.FC = () => {
  const t = useTime();
  const zoom = between(t, 31.6, 32.6, 1, 1.28) * between(t, 32.9, 33.3, 1, 1 / 1.28) * between(t, 34.3, 35.1, 1, 1.28);
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
      <Window url="localhost · Image lab" width={1520} height={855}>
        {t < 33.0 ? <Clip name="lab-real" trim={0.2} rate={1.5} zoom={zoom} x={0.86} y={0.42}/> : <Clip name="lab-ai" trim={0.5} rate={1.5} zoom={zoom} x={0.86} y={0.42}/>}
      </Window>
      <Reveal at={30.9} style={{position: 'absolute', top: 30}}><Kicker>Image lab</Kicker></Reveal>
    </AbsoluteFill>
  );
};

const Node: React.FC<{at: number; title: string; sub: string; children: React.ReactNode}> = ({at, title, sub, children}) => (
  <Reveal at={at} style={{width: 360, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22}}>
    <div style={{width: 220, height: 160, borderRadius: 24, background: color.stageLift, border: `1.5px solid ${color.line}`, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden'}}>{children}</div>
    <div style={{fontFamily: FONT, fontSize: 34, fontWeight: 600, color: color.text}}>{title}</div>
    <div style={{fontFamily: FONT, fontSize: 24, color: color.muted, marginTop: -12}}>{sub}</div>
  </Reveal>
);

const Laptop: React.FC = () => (
  <svg width="130" height="90" viewBox="0 0 130 90" fill="none">
    <rect x="18" y="6" width="94" height="62" rx="6" stroke={color.lavender} strokeWidth="4"/>
    <path d="M4 80h122" stroke={color.lavender} strokeWidth="5" strokeLinecap="round"/>
    <path d="M44 38l12 10 26-24" stroke={color.tealBright} strokeWidth="5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const PACKET_START = 39.0;
const PACKET_PERIOD = 2.2;
// Scores these images actually received in the recordings.
const SCANS = [
  {image: 't-crab', score: 0.0},
  {image: 't-lion', score: 62.0},
  {image: 't-strawberry', score: 11.4},
  {image: 't-pinup', score: 90.9},
  {image: 't-street', score: 0.1},
];

const packetIndex = (t: number) => Math.max(0, Math.floor((t - PACKET_START) / PACKET_PERIOD));

const LiveScore: React.FC = () => {
  const t = useTime();
  const arrivals = Math.floor((t - PACKET_START) / PACKET_PERIOD);
  if (arrivals < 1) return <div style={{fontFamily: FONT, fontSize: 52, fontWeight: 700, color: color.muted}}>—</div>;
  const {score} = SCANS[(arrivals - 1) % SCANS.length];
  const since = t - PACKET_START - arrivals * PACKET_PERIOD;
  const pop = 1 + 0.18 * Math.max(0, 1 - since / 0.35);
  const flagged = score >= 50;
  const tone = flagged ? color.lavender : color.tealBright;
  return (
    <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, transform: `scale(${pop})`}}>
      <div style={{fontFamily: FONT, fontSize: 52, fontWeight: 700, color: tone, fontVariantNumeric: 'tabular-nums'}}>{score.toFixed(1)}%</div>
      <div style={{fontFamily: FONT, fontSize: 20, fontWeight: 600, letterSpacing: 2, textTransform: 'uppercase', color: tone}}>{flagged ? 'AI signal' : 'Clear'}</div>
    </div>
  );
};

export const Privacy: React.FC = () => {
  const t = useTime();
  const terminalOut = between(t, 37.9, 38.5, 0, 1);
  const packet = (t - PACKET_START) % PACKET_PERIOD / PACKET_PERIOD;
  const flowing = t > PACKET_START;
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', opacity: 1 - terminalOut, transform: `translateY(${-terminalOut * 60}px)`}}>
        <Reveal at={36.0}><Headline size={88}>Everything runs locally.</Headline></Reveal>
        <Reveal at={36.3} style={{marginTop: 60}}>
          <Window url="Terminal" width={1300} height={218}><Clip name="terminal" rate={0.75}/></Window>
        </Reveal>
      </AbsoluteFill>
      <AbsoluteFill style={{opacity: terminalOut}}>
        <div style={{position: 'absolute', left: 200, right: 200, top: 150, height: 560, borderRadius: 36, border: `2px dashed ${color.lavender}66`}}/>
        <div style={{position: 'absolute', left: 240, top: 130, padding: '6px 16px', background: color.stage, fontFamily: FONT, fontSize: 26, fontWeight: 600, letterSpacing: 3, color: color.lavender, textTransform: 'uppercase'}}>Your device</div>
        <div style={{position: 'absolute', left: 200, right: 200, top: 240, display: 'flex', justifyContent: 'space-between', padding: '0 80px'}}>
          <Node at={38.1} title="Image on the page" sub="read from your browser">
            <Img src={staticFile(`stills/${SCANS[packetIndex(t) % SCANS.length].image}.png`)} style={{width: '100%', height: '100%', objectFit: 'cover'}}/>
          </Node>
          <Node at={38.8} title="Local server" sub="127.0.0.1 on your laptop"><Laptop/></Node>
          <Node at={39.6} title="Score" sub="shown on the image">
            <LiveScore/>
          </Node>
        </div>
        {flowing ? (
          <div style={{position: 'absolute', top: 318, left: 520 + packet * 880, width: 18, height: 18, borderRadius: 9, background: color.lavender, boxShadow: `0 0 24px ${color.lavender}`, opacity: Math.min(1, packet * 6, (1 - packet) * 20)}}/>
        ) : null}
        <div style={{position: 'absolute', left: 0, right: 0, bottom: 150, display: 'flex', justifyContent: 'center', gap: 24}}>
          <Reveal at={41.3}><Chip tone="purple">Lightweight</Chip></Reveal>
          <Reveal at={42.85}><Chip tone="teal">No account</Chip></Reveal>
          <Reveal at={43.75}><Chip tone="teal">No API key</Chip></Reveal>
          <Reveal at={44.7}><Chip tone="teal">Nothing leaves your device</Chip></Reveal>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};