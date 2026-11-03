import {isRedditImageUrl, MAX_BYTES} from '../detection/model';

export type ImageSource = 'preview-first' | 'prefer-original' | 'displayed';
export type PostContext = {postUrl?: string; subreddit?: string; candidates?: string[]};
export type ImageInput = {blob: Blob; sourceUrl: string; displayedUrl: string; inputSource: 'original' | 'displayed'; contentType: string; downloadBytes: number; downloadMs: number; fallbackReason?: string};

// A preview URL is not a guaranteed original URL. Only native Reddit filenames
// are eligible for a best-effort rewrite; external previews remain untouched.
export function originalCandidate(displayed: string, linked?: string | string[]): string | undefined {
  if (!isRedditImageUrl(displayed)) return;
  const url = new URL(displayed);
  if (url.hostname === 'i.redd.it') return url.href;
  if (url.hostname !== 'preview.redd.it') return;
  const plain = /^\/([a-z0-9]{6,32})\.(png|jpe?g|webp|avif)$/i.exec(url.pathname);
  const slug = /^\/[^/]+-v0-([a-z0-9]{6,32})\.(png|jpe?g|webp|avif)$/i.exec(url.pathname);
  const match = plain ?? slug;
  if (!match) return;
  for (const candidate of typeof linked === 'string' ? [linked] : linked ?? []) {
    const resolved = redditMediaUrl(candidate);
    if (!resolved) continue;
    const target = new URL(resolved);
    if (target.hostname === 'i.redd.it' && target.pathname.split('.')[0].toLowerCase() === `/${match[1].toLowerCase()}`) return target.href;
  }
  // Caption variants may contain additional pixels; never infer their originals.
  if (!plain && url.pathname.startsWith('/caption-v0-')) return;
  return `https://i.redd.it/${match[1]}.${match[2]}`;
}

export function redditMediaUrl(value: string): string | undefined {
  try {
    const url = new URL(value);
    if (isRedditImageUrl(url.href)) return url.href;
    if (url.protocol === 'https:' && /^(www\.|old\.|new\.)?reddit\.com$/.test(url.hostname) && url.pathname === '/media') {
      const nested = url.searchParams.get('url');
      if (nested && isRedditImageUrl(nested)) return nested;
    }
  } catch {}
}

export function postElement(image: Element): Element | null {
  let node: Element | null=image;
  while(node) {
    const found=node.closest('shreddit-post, .thing.link, [data-testid="post-container"], article');
    if(found)return found;
    const root=node.getRootNode();node=root instanceof ShadowRoot ? root.host : null;
  }
  return null;
}

export function postContext(image: HTMLImageElement): PostContext {
  const post=postElement(image);if(!post)return {};
  const candidates: string[]=[];
  const add=(value:string|null)=>{if(value && value.length<4096 && candidates.length<128)candidates.push(value);};
  add(post.getAttribute('content-href'));add(post.getAttribute('data-url'));add(image.closest('a')?.href ?? null);
  const roots: (Element|ShadowRoot)[]=[post];
  for(let i=0;i<roots.length && i<32;i++) {
    const root=roots[i];
    for(const anchor of Array.from(root.querySelectorAll<HTMLAnchorElement>('a[href]')).slice(0,64))add(anchor.href);
    // Reddit galleries expose original sources on lazy/hidden image elements.
    // Matching by media ID below prevents selecting another gallery frame.
    for(const media of Array.from(root.querySelectorAll<HTMLImageElement>('img[src]')).slice(0,64)) {
      if(isRedditImageUrl(media.src) && new URL(media.src).hostname==='i.redd.it')add(media.src);
    }
    for(const child of Array.from(root.querySelectorAll('*')).slice(0,1000))if(child.shadowRoot)roots.push(child.shadowRoot);
  }
  // Known DOM metadata attributes only; never evaluate scripts or page code.
  const strings=(value:unknown,depth=0)=>{
    if(depth>6 || candidates.length>=128)return;
    if(typeof value==='string')add(value);
    else if(Array.isArray(value))value.slice(0,64).forEach(v=>strings(v,depth+1));
    else if(value && typeof value==='object')Object.values(value).slice(0,64).forEach(v=>strings(v,depth+1));
  };
  for(const name of ['media-metadata','media_metadata','data-media-metadata']) {
    const value=post.getAttribute(name);if(value && value.length<200000)try {strings(JSON.parse(value));}catch {}
  }
  let postUrl=post.getAttribute('permalink') ?? post.getAttribute('data-permalink');
  if(!postUrl)postUrl=Array.from(post.querySelectorAll<HTMLAnchorElement>('a[href]')).find(a=>/\/comments\//.test(a.pathname))?.href ?? null;
  try {if(postUrl)postUrl=new URL(postUrl,location.origin).href;}catch {postUrl=null;}
  const subreddit=post.getAttribute('subreddit-prefixed-name') ?? post.getAttribute('subreddit-name') ?? post.getAttribute('data-subreddit') ?? undefined;
  return {postUrl:postUrl ?? (/\/comments\//.test(location.pathname)?location.href:undefined),subreddit,candidates};
}

export class ImageFetchError extends Error {
  constructor(message:string,public status?:number){super(message);}
}

export async function fetchImage(url: string, fetcher: typeof fetch = fetch): Promise<{blob: Blob; url: string;status:number}> {
  if (!isRedditImageUrl(url)) throw new Error('Unsupported image host.');
  const response = await fetcher(url, {credentials:'omit',referrerPolicy:'no-referrer',signal:AbortSignal.timeout(20000)});
  if (!response.ok || !isRedditImageUrl(response.url)) throw new ImageFetchError(`Reddit image fetch failed (${response.status}).`,response.status);
  const type = response.headers.get('content-type')?.split(';')[0].trim() ?? '';
  if (!['image/jpeg','image/png','image/webp','image/avif'].includes(type)) throw new Error('Only still JPEG, PNG, WebP and AVIF images are supported.');
  if (Number(response.headers.get('content-length')) > MAX_BYTES) {await response.body?.cancel(); throw new Error('Image exceeds 20 MB.');}
  const reader=response.body?.getReader();if(!reader)throw new Error('Empty image response.');
  const chunks:Uint8Array<ArrayBuffer>[]=[];let total=0;
  try {
    for (;;) {
      const {done,value}=await reader.read();if(done)break;
      total+=value.byteLength;
      if(total>MAX_BYTES)throw new Error('Image exceeds 20 MB.');
      chunks.push(new Uint8Array(value));
    }
  } catch(error) {await reader.cancel().catch(()=>{});throw error;}
  finally {reader.releaseLock();}
  if (!total) throw new Error('Empty image response.');
  return {blob:new Blob(chunks,{type}),url:response.url,status:response.status};
}

export async function imageInput(displayedUrl: string, source: ImageSource, linkedOriginal?: string, fetcher: typeof fetch = fetch): Promise<ImageInput> {
  const started=performance.now();
  const original=source==='prefer-original' ? originalCandidate(displayedUrl,linkedOriginal) : undefined;
  let fallbackReason:string|undefined;
  if (original && original!==displayedUrl) {
    try {
      const {blob,url}=await fetchImage(original,fetcher);
      return {blob,sourceUrl:url,displayedUrl,inputSource:new URL(url).hostname==='i.redd.it'?'original':'displayed',contentType:blob.type,downloadBytes:blob.size,downloadMs:Math.round(performance.now()-started)};
    } catch(error) {fallbackReason=error instanceof Error ? error.message : String(error);}
  }
  const {blob,url}=await fetchImage(displayedUrl,fetcher);
  return {blob,sourceUrl:url,displayedUrl,inputSource:new URL(displayedUrl).hostname==='i.redd.it'?'original':'displayed',
    contentType:blob.type,downloadBytes:blob.size,downloadMs:Math.round(performance.now()-started),...(fallbackReason?{fallbackReason}:{})};
}
