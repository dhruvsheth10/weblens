import React from 'react';
import {AbsoluteFill, Img, interpolate, random, staticFile} from 'remotion';
import {Chip, Clip, Headline, Kicker, LensMark, Reveal, Ring, useTime, Window} from './components';
import {between, color, FONT, rise} from './theme';

const format = (value: number, decimals = 0) => value.toLocaleString('en-US', {minimumFractionDigits: decimals, maximumFractionDigits: decimals});

// The settings recording selects Sherlock 2 at ~1.8 s, briefly passes through Sherlock 1.0, and settles on Sherlock 2 at 3.8 s.
const MODEL_CUT = 48.4;

export const ModelPick: React.FC = () => {
  const t = useTime();
  return (
    <AbsoluteFill style={{flexDirection: 'row', alignItems: 'center', padding: '0 100px', gap: 70}}>
      <div style={{width: 610}}>
        <Reveal at={46.5}><Kicker>The detector</Kicker></Reveal>
        <Reveal at={47.9} style={{marginTop: 20}}><Headline size={120}>Sherlock 2</Headline></Reveal>
        <Reveal at={48.8} style={{marginTop: 26}}>
          <div style={{fontFamily: FONT, fontSize: 34, lineHeight: 1.4, color: color.muted}}>A vision transformer I fine-tuned on Reddit images</div>
        </Reveal>
      </div>
      <Reveal at={46.4}>
        <Window url="localhost · Settings" width={1040} height={585}>
          {t < MODEL_CUT ? <Clip name="models"/> : <Clip name="models" trim={3.8 - (MODEL_CUT - 46.0) * 0.9} rate={0.9}/>}
        </Window>
      </Reveal>
    </AbsoluteFill>
  );
};

export const Collected: React.FC = () => {
  const t = useTime();
  const count = Math.round(interpolate(t, [50.1, 51.7], [0, 229044], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: (x) => 1 - Math.pow(1 - x, 3)}));
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', textAlign: 'center'}}>
      <Reveal at={50.0}><Headline size={220} style={{fontVariantNumeric: 'tabular-nums'}}>{format(count)}</Headline></Reveal>
      <Reveal at={50.6} style={{marginTop: 4}}><div style={{fontFamily: FONT, fontSize: 48, color: color.muted}}>Reddit images collected</div></Reveal>
      <div style={{display: 'flex', gap: 40, marginTop: 70, perspective: 1200}}>
        <LabelCard at={52.4} kicker="When" text="Post date"/>
        <LabelCard at={53.2} kicker="Where" text="Community rules"/>
      </div>
    </AbsoluteFill>
  );
};

/** Flips up like a tag, then types its label out. */
const LabelCard: React.FC<{at: number; kicker: string; text: string}> = ({at, kicker, text}) => {
  const t = useTime();
  const flip = rise(t, at, 0.7);
  const typed = Math.floor(interpolate(t, [at + 0.35, at + 0.35 + text.length * 0.04], [0, text.length], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));
  const caret = typed < text.length || Math.floor(t * 2.5) % 2 === 0;
  return (
    <div style={{width: 400, padding: '26px 34px', borderRadius: 22, background: 'rgba(185,168,255,0.08)', border: `1.5px solid ${color.lavender}66`, textAlign: 'left', opacity: flip, transform: `rotateX(${(1 - flip) * -95}deg)`, transformOrigin: '50% 0'}}>
      <div style={{fontFamily: FONT, fontSize: 22, fontWeight: 600, letterSpacing: 4, textTransform: 'uppercase', color: color.lavender}}>{kicker}</div>
      <div style={{fontFamily: FONT, fontSize: 46, fontWeight: 600, color: color.text, marginTop: 8, whiteSpace: 'nowrap'}}>
        {text.slice(0, typed)}<span style={{opacity: caret ? 1 : 0, color: color.lavender}}>|</span>
      </div>
    </div>
  );
};

const COLS = 20;
const ROWS = 10;
const HUES = ['#b9a8ff', '#5fd3c0', '#f2b880', '#f07c9a', '#8fb8ff', '#d6e685', '#c49bf2', '#7fe0e8', '#ffd76e', '#ff9f80', '#a3e0a8', '#e0a8d8'];

export const HeldOut: React.FC = () => {
  const t = useTime();
  const dots = Array.from({length: COLS * ROWS}, (_, i) => {
    const community = Math.floor(random(`c${i}`) * 12);
    const unseenCommunity = community < 3;
    const unseenAuthor = !unseenCommunity && random(`a${i}`) < 0.16;
    return {i, community, held: unseenCommunity || unseenAuthor, unseenCommunity};
  });
  let heldIndex = 0;
  return (
    <AbsoluteFill>
      <div style={{position: 'absolute', top: 90, left: 0, right: 0, textAlign: 'center'}}>
        <Reveal at={54.8}><Headline size={70}>Tested only on what it never saw</Headline></Reveal>
      </div>
      {[{left: 130, label: 'Training', at: 54.9}, {left: 1070, label: 'Test', at: 55.4}].map((box) => (
        <Reveal key={box.label} at={box.at} style={{position: 'absolute', left: box.left, top: 250, width: 720, height: 540, borderRadius: 32, background: 'rgba(255,255,255,0.03)', border: `1.5px solid ${color.line}`}}>
          <div style={{position: 'absolute', top: 24, left: 32, fontFamily: FONT, fontSize: 26, fontWeight: 600, letterSpacing: 3, textTransform: 'uppercase', color: color.muted}}>{box.label}</div>
        </Reveal>
      ))}
      {dots.map((dot) => {
        const homeX = 170 + (dot.i % COLS) * 33;
        const homeY = 330 + Math.floor(dot.i / COLS) * 42;
        let x = homeX;
        let y = homeY;
        if (dot.held) {
          const slot = heldIndex++;
          const start = (dot.unseenCommunity ? 57.0 : 56.1) + (slot % 12) * 0.03;
          const p = between(t, start, start + 0.9, 0, 1);
          x = interpolate(p, [0, 1], [homeX, 1120 + (slot % 16) * 38]);
          y = interpolate(p, [0, 1], [homeY, 330 + Math.floor(slot / 16) * 46]);
        }
        const appear = rise(t, 54.9 + (dot.i % COLS) * 0.012, 0.5);
        return <div key={dot.i} style={{position: 'absolute', left: x, top: y, width: 18, height: 18, borderRadius: 9, background: HUES[dot.community], opacity: appear * (dot.held ? 1 : 0.75)}}/>;
      })}
      <div style={{position: 'absolute', bottom: 120, left: 0, right: 0, display: 'flex', justifyContent: 'center', gap: 24}}>
        <Reveal at={56.1}><Chip tone="plain">Unseen people</Chip></Reveal>
        <Reveal at={56.9}><Chip tone="plain">12 unseen communities</Chip></Reveal>
        <Reveal at={58.7}><Chip tone="teal">No memorized answers</Chip></Reveal>
      </div>
    </AbsoluteFill>
  );
};

type Stat = {value: number; decimals: number; suffix: string; label: string; detail: string; at: number; focus: number};
const STATS: Stat[] = [
  {value: 229044, decimals: 0, suffix: '', label: 'Reddit images collected', detail: 'labeled by community and date', at: 61.2, focus: 0},
  {value: 24708, decimals: 0, suffix: '', label: 'test posts from unseen authors', detail: '+8,185 from 12 unseen communities', at: 61.8, focus: 64.8},
  {value: 58, decimals: 0, suffix: '%', label: 'of AI images caught', detail: '57.6% on the author-held-out test', at: 62.4, focus: 68.7},
  {value: 0.15, decimals: 2, suffix: '%', label: 'of real images flagged', detail: '31 of 20,699 real images', at: 63.0, focus: 71.9},
];

export const Stats: React.FC = () => {
  const t = useTime();
  const active = [...STATS].reverse().find((stat) => stat.focus > 0 && t >= stat.focus);
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
      <Reveal at={61.0} style={{position: 'absolute', top: 150}}><Kicker>Results on posts it never saw</Kicker></Reveal>
      <div style={{display: 'flex', gap: 30}}>
        {STATS.map((stat) => {
          const count = interpolate(t, [stat.at, stat.at + 1.2], [0, stat.value], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: (x) => 1 - Math.pow(1 - x, 3)});
          const isActive = active === stat;
          const dim = active && !isActive ? 0.45 : 1;
          const lift = rise(t, stat.focus || 999, 0.4);
          return (
            <Reveal key={stat.label} at={stat.at} style={{opacity: dim}}>
              <div style={{width: 400, height: 360, borderRadius: 32, padding: 40, boxSizing: 'border-box', background: isActive ? 'rgba(185,168,255,0.10)' : 'rgba(255,255,255,0.04)', border: `2px solid ${isActive ? color.lavender : color.line}`, transform: `translateY(${isActive ? -18 * lift : 0}px)`, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end'}}>
                <div style={{fontFamily: FONT, fontSize: stat.value > 1000 ? 92 : 120, fontWeight: 600, letterSpacing: -3, color: isActive ? color.lavender : color.text, fontVariantNumeric: 'tabular-nums'}}>{format(count, stat.decimals)}{stat.suffix}</div>
                <div style={{fontFamily: FONT, fontSize: 30, color: color.text, marginTop: 8}}>{stat.label}</div>
                <div style={{fontFamily: FONT, fontSize: 22, color: color.muted, marginTop: 12}}>{stat.detail}</div>
              </div>
            </Reveal>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

const Marker: React.FC<{at: number; score: number; image: string; flagged: boolean}> = ({at, score, image, flagged}) => {
  const p = rise(useTime(), at, 0.6);
  const tone = flagged ? color.lavender : color.tealBright;
  return (
    <div style={{position: 'absolute', left: `${score}%`, bottom: 70, transform: `translate(-50%, ${(1 - p) * -40}px)`, opacity: p, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14}}>
      <Img src={staticFile(`stills/${image}.png`)} style={{width: 170, height: 170, objectFit: 'cover', borderRadius: 20, border: `3px solid ${tone}`}}/>
      <div style={{fontFamily: FONT, fontSize: 34, fontWeight: 700, color: tone}}>{score.toFixed(1)}%</div>
      <div style={{fontFamily: FONT, fontSize: 22, color: color.muted}}>{flagged ? 'flagged' : 'left alone'}</div>
      <div style={{width: 3, height: 26, background: tone}}/>
    </div>
  );
};

export const Careful: React.FC = () => {
  const t = useTime();
  const fill = between(t, 74.2, 75.4, 0, 1);
  return (
    <AbsoluteFill style={{alignItems: 'center'}}>
      <Reveal at={73.8} style={{marginTop: 130}}><Headline size={76}>Quiet unless it's confident</Headline></Reveal>
      <div style={{position: 'absolute', left: 260, right: 260, top: 300, height: 520}}>
        <Marker at={75.0} score={11.4} image="t-strawberry" flagged={false}/>
        <Marker at={75.6} score={62.0} image="t-lion" flagged/>
        <div style={{position: 'absolute', left: 0, right: 0, bottom: 40, height: 14, borderRadius: 7, background: 'rgba(255,255,255,0.08)'}}/>
        <div style={{position: 'absolute', left: 0, width: `${fill * 100}%`, bottom: 40, height: 14, borderRadius: 7, background: `linear-gradient(90deg, ${color.tealBright} 0%, ${color.tealBright} 50%, ${color.lavender} 50%)`}}/>
        <div style={{position: 'absolute', left: '50%', bottom: 0, height: 96, width: 3, background: color.text, opacity: fill}}/>
        <div style={{position: 'absolute', left: '50%', bottom: -56, transform: 'translateX(-50%)', fontFamily: FONT, fontSize: 26, color: color.text, opacity: fill, whiteSpace: 'nowrap'}}>Flag cutoff</div>
      </div>
      <Reveal at={77.0} style={{position: 'absolute', bottom: 90}}><Chip tone="teal">A false accusation costs more than a miss</Chip></Reveal>
    </AbsoluteFill>
  );
};

export const Art: React.FC = () => (
  <AbsoluteFill style={{flexDirection: 'row', alignItems: 'center', padding: '0 120px', gap: 90}}>
    <Reveal at={80.7}>
      <Window url="reddit.com/r/aiArt" width={760} height={590}>
        <Img src={staticFile('stills/art-miss.png')} style={{width: '100%', height: '100%', objectFit: 'cover'}}/>
        <Ring at={82.6} left={12} top={13} width={248} height={57} tone="#f2b880"/>
      </Window>
    </Reveal>
    <div style={{flex: 1}}>
      <Reveal at={81.0}><Kicker style={{color: '#f2b880'}}>Known limit</Kicker></Reveal>
      <Reveal at={81.7} style={{marginTop: 20}}><Headline size={86}>Art is still its hardest case.</Headline></Reveal>
      <Reveal at={83.4} style={{marginTop: 34}}>
        <div style={{fontFamily: FONT, fontSize: 36, lineHeight: 1.45, color: color.muted}}>This AI-made artwork scored 37% and slipped through. A flag is a reason to look closer.</div>
      </Reveal>
    </div>
  </AbsoluteFill>
);

const TOOLS = [
  {at: 87.6, names: 'Python · PyTorch', role: 'Trains the model'},
  {at: 89.9, names: 'FastAPI', role: 'Serves it locally'},
  {at: 92.7, names: 'TypeScript · React', role: 'Extension and image lab'},
];

export const Tools: React.FC = () => (
  <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
    <Reveal at={86.7}><Kicker>How it's built</Kicker></Reveal>
    <div style={{display: 'flex', alignItems: 'center', gap: 28, marginTop: 60}}>
      {TOOLS.map((tool, i) => (
        <React.Fragment key={tool.names}>
          {i > 0 ? <Reveal at={tool.at - 0.2}><div style={{fontFamily: FONT, fontSize: 60, color: color.lavender}}>→</div></Reveal> : null}
          <Reveal at={tool.at}>
            <div style={{width: 470, height: 260, borderRadius: 30, padding: 44, boxSizing: 'border-box', background: 'rgba(255,255,255,0.04)', border: `1.5px solid ${color.line}`, display: 'flex', flexDirection: 'column', justifyContent: 'space-between'}}>
              <div style={{fontFamily: FONT, fontSize: 26, color: color.muted}}>{tool.role}</div>
              <div style={{fontFamily: FONT, fontSize: 50, fontWeight: 600, color: color.text, letterSpacing: -1}}>{tool.names}</div>
            </div>
          </Reveal>
        </React.Fragment>
      ))}
    </div>
  </AbsoluteFill>
);

const SITES = [
  {name: 'News sites', x: -560, y: -170},
  {name: 'Social feeds', x: 560, y: -170},
  {name: 'Search results', x: -600, y: 170},
  {name: 'Blogs', x: 600, y: 170},
  {name: 'Shopping', x: 0, y: 290},
];

export const Future: React.FC = () => {
  const t = useTime();
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
      <Reveal at={98.0} style={{position: 'absolute', top: 120}}><Headline size={72}>Next: the rest of the web</Headline></Reveal>
      <Reveal at={95.1}>
        <div style={{display: 'flex', alignItems: 'center', gap: 22, padding: '30px 46px', borderRadius: 26, background: 'rgba(95,211,192,0.10)', border: `2px solid ${color.tealBright}`}}>
          <div style={{fontFamily: FONT, fontSize: 56, fontWeight: 600, color: color.text}}>reddit.com</div>
          <div style={{fontFamily: FONT, fontSize: 26, fontWeight: 600, color: color.tealBright, letterSpacing: 2, textTransform: 'uppercase'}}>Tested</div>
        </div>
      </Reveal>
      {SITES.map((site, i) => {
        const p = rise(t, 97.6 + i * 0.25, 0.7);
        return (
          <div key={site.name} style={{position: 'absolute', left: '50%', top: '50%', transform: `translate(calc(-50% + ${site.x * p}px), calc(-50% + ${site.y * p}px))`, opacity: p, padding: '20px 34px', borderRadius: 22, border: `2px dashed ${color.lavender}88`, fontFamily: FONT, fontSize: 38, color: color.text}}>{site.name}</div>
        );
      })}
    </AbsoluteFill>
  );
};

export const Close: React.FC = () => {
  const t = useTime();
  const mark = rise(t, 101.9, 1.0);
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', textAlign: 'center'}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 40, opacity: mark, transform: `scale(${0.92 + 0.08 * mark})`}}>
        <LensMark size={150} progress={mark}/>
        <Headline size={170}>WebLens</Headline>
      </div>
      <Reveal at={103.1} style={{marginTop: 40}}>
        <div style={{fontFamily: FONT, fontSize: 52, color: color.lavender}}>Know exactly what you're looking at.</div>
      </Reveal>
      <Reveal at={105.0} style={{marginTop: 90}}>
        <div style={{fontFamily: FONT, fontSize: 30, color: color.muted, letterSpacing: 1}}>Built by Dhruv</div>
      </Reveal>
    </AbsoluteFill>
  );
};
