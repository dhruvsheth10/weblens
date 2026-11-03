// Reddit reuses image elements while galleries and feed posts change.
export function sameImage(image: Pick<HTMLImageElement,'isConnected'|'currentSrc'|'src'>, url:string) {
  return image.isConnected && (image.currentSrc || image.src) === url;
}
