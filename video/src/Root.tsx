import React from 'react';
import {Composition} from 'remotion';
import {DURATION, WebLensDemo} from './Video';
import {FPS, HEIGHT, sec, WIDTH} from './theme';

export const Root: React.FC = () => (
  <Composition id="WebLensDemo" component={WebLensDemo} durationInFrames={sec(DURATION)} fps={FPS} width={WIDTH} height={HEIGHT}/>
);
