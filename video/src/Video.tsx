import React from 'react';
import {AbsoluteFill, Audio, interpolate, Sequence, staticFile, useCurrentFrame} from 'remotion';
import {Scene, Stage} from './components';
import {Audience, FeedAi, FeedBrowse, FeedClear, FeedFlag, Hook, Intro, Lab, Privacy} from './scenesProduct';
import {Art, Careful, Close, Collected, Future, HeldOut, ModelPick, Stats, Tools} from './scenesStory';
import {color, FPS, sec} from './theme';

export const DURATION = 109;
/** Part 2 of the narration starts here on the global timeline. */
const PART2 = 45.9;

const clampMix = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

// The track grows about 12 dB louder over its first 90 s, so the bed under the voice tapers to stay level.
// It swells in the stats pause and under the end card.
const musicVolume = (frame: number) => {
  const t = frame / FPS;
  const bed = interpolate(t, [0, 15, 40, 70, 100], [0.3, 0.22, 0.14, 0.1, 0.1], clampMix);
  const pause = interpolate(t, [60.8, 61.6, 64.0, 64.9], [1, 3.2, 3.2, 1], clampMix);
  const outro = interpolate(t, [104.4, 105.6], [1, 3.4], clampMix);
  const fade = interpolate(t, [0, 0.8, DURATION - 3, DURATION], [0, 1, 1, 0], {...clampMix, easing: (x) => x * x * (3 - 2 * x)});
  return bed * pause * outro * fade;
};

export const WebLensDemo: React.FC = () => {
  const frame = useCurrentFrame();
  const blackout = interpolate(frame / FPS, [DURATION - 1.4, DURATION - 0.2], [0, 1], clampMix);
  return (
    <AbsoluteFill style={{background: color.stage}}>
      <Stage/>
      <Scene start={0} end={8.4}><Hook/></Scene>
      <Scene start={8.4} end={16.0}><Intro/></Scene>
      <Scene start={16.0} end={20.3}><Audience/></Scene>
      <Scene start={20.3} end={22.4}><FeedBrowse/></Scene>
      <Scene start={22.4} end={25.0}><FeedClear/></Scene>
      <Scene start={25.0} end={27.5}><FeedAi/></Scene>
      <Scene start={27.5} end={30.5}><FeedFlag/></Scene>
      <Scene start={30.5} end={35.95}><Lab/></Scene>
      <Scene start={35.95} end={46.3}><Privacy/></Scene>
      <Scene start={46.3} end={50.0}><ModelPick/></Scene>
      <Scene start={50.0} end={54.6}><Collected/></Scene>
      <Scene start={54.6} end={60.9}><HeldOut/></Scene>
      <Scene start={60.9} end={73.6}><Stats/></Scene>
      <Scene start={73.6} end={80.4}><Careful/></Scene>
      <Scene start={80.4} end={86.4}><Art/></Scene>
      <Scene start={86.4} end={94.8}><Tools/></Scene>
      <Scene start={94.8} end={101.5}><Future/></Scene>
      <Scene start={101.5} end={DURATION + 0.3}><Close/></Scene>
      <AbsoluteFill style={{background: '#000', opacity: blackout}}/>
      <Audio src={staticFile('audio/part1.wav')}/>
      <Sequence from={sec(PART2)}><Audio src={staticFile('audio/part2.wav')}/></Sequence>
      <Audio src={staticFile('audio/music.wav')} volume={musicVolume}/>
    </AbsoluteFill>
  );
};
