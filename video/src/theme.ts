import {continueRender, delayRender, Easing, interpolate, staticFile} from 'remotion';

export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;

export const color = {
  stage: '#121119',
  stageLift: '#1b1a26',
  ink: '#272638',
  paper: '#f8f9fc',
  text: '#f4f3fa',
  muted: '#a19fb6',
  purple: '#6550ad',
  lavender: '#b9a8ff',
  lavenderSoft: '#eeebfa',
  teal: '#14776f',
  tealBright: '#5fd3c0',
  line: 'rgba(255,255,255,0.10)',
};

export const FONT = 'Google Sans, system-ui, sans-serif';

const fontHandle = delayRender('Loading Google Sans');
const face = new FontFace('Google Sans', `url(${staticFile('fonts/google-sans-latin.woff2')}) format('woff2')`, {weight: '400 700'});
face.load().then(() => {
  document.fonts.add(face);
  continueRender(fontHandle);
}).catch((error) => {
  console.error(error);
  continueRender(fontHandle);
});

export const sec = (seconds: number) => Math.round(seconds * FPS);

const smooth = Easing.bezier(0.16, 1, 0.3, 1);

/** 0→1 progress that starts at `at` seconds and settles over `duration`. */
export const rise = (t: number, at: number, duration = 0.6) =>
  interpolate(t, [at, at + duration], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: smooth});

export const between = (t: number, from: number, to: number, a: number, b: number) =>
  interpolate(t, [from, to], [a, b], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.inOut(Easing.cubic)});
