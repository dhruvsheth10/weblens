import type {ImageMark,PageStatus} from './presentation';

export const OVERLAY_ID='reddit-lens-overlay-root';
const CSS=`
:host{all:initial;font-family:'Reddit Lens Google Sans','Google Sans',sans-serif;color:#272638;color-scheme:light}
*{box-sizing:border-box}.layer{position:fixed;inset:0;z-index:2147483646;pointer-events:none}
.frame{position:fixed;border:3px solid #9188b8;border-radius:8px;pointer-events:none;overflow:hidden;--ink:#53408f;--fill:#f2effc;--edge:#c9c0e5}
.frame[data-state=ai]{border-color:#7355bc;--ink:#fff;--fill:#6550ad;--edge:#6550ad}
.frame[data-state=uncertain]{border-color:#b89137;--ink:#765615;--fill:#fcf4df;--edge:#ddc88f}
.frame[data-state=unmarked]{border-color:#26978b;--ink:#14685f;--fill:#e6f5f1;--edge:#a4d6cb}
.frame[data-state=error]{border-color:#b55068;--ink:#94354b;--fill:#fceef1;--edge:#e4b9c4}
.badge{position:absolute;left:9px;top:9px;max-width:calc(100% - 18px);min-width:0;padding:9px 12px;border:1px solid var(--edge);border-radius:8px;background:var(--fill);color:var(--ink);box-shadow:0 2px 9px #27263824;font-size:12px;line-height:1.4;pointer-events:auto}
.badge-head{display:flex;align-items:center;gap:9px}.badge b{font-weight:600}.score{margin-left:auto;font-weight:700;white-space:nowrap;font-variant-numeric:tabular-nums}.badge small{display:block;margin-top:3px;font-size:10px}.badge small:empty{display:none}
.spinner{display:none;width:12px;height:12px;flex-shrink:0;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;animation:lens-spin .85s linear infinite}.frame[data-busy=true] .spinner{display:inline-block}
.panel{position:fixed;right:18px;bottom:18px;width:280px;max-width:calc(100vw - 24px);padding:16px;background:#fcfcff;color:#272638;border:1px solid #dedbea;border-radius:14px;box-shadow:0 6px 28px #30294420;pointer-events:auto;font-size:12px;line-height:1.5}
.heading{display:flex;align-items:center;gap:8px;font-size:13px}.heading strong{font-weight:600;letter-spacing:-.2px}.state{margin-left:auto;font-size:10px;font-weight:500;color:#14776f}
.dot{width:7px;height:7px;background:#26978b;border-radius:50%;flex-shrink:0}.panel[data-mode=scanning] .dot,.panel[data-mode=checking] .dot,.panel[data-mode=starting] .dot{width:11px;height:11px;border:2px solid #b7ded5;border-top-color:#14776f;background:transparent;animation:lens-spin .85s linear infinite}
.panel[data-mode=error] .dot{background:#a13850}.panel[data-mode=error] .state{color:#a13850}.panel[data-mode=paused] .dot,.panel[data-mode=scope] .dot{background:#858198}.panel[data-mode=paused] .state,.panel[data-mode=scope] .state{color:#656579}
.model{display:flex;align-items:baseline;gap:7px;margin-top:12px;font-size:11px;color:#6550ad}.model small{color:#656579;font-size:10px}.model b{font-weight:500}
.detail{margin:6px 0 0;color:#656579;font-size:11px;overflow-wrap:anywhere}.counts{display:flex;gap:14px;border-top:1px solid #e3e3ed;padding-top:11px;margin-top:12px;color:#656579;font-size:10px}.counts b{font-weight:600;color:#272638;font-variant-numeric:tabular-nums}
.actions{display:flex;gap:7px;margin-top:13px}.actions button{font:500 11px 'Reddit Lens Google Sans','Google Sans',sans-serif;border:1px solid #dedbea;border-radius:7px;padding:7px 12px;background:#fff;color:#6550ad;cursor:pointer;transition:background .15s}.actions button:hover{background:#eeebfa}.actions .primary{background:#eeebfa;color:#53408f;border-color:#eeebfa;margin-right:auto}.actions button:focus-visible{outline:2px solid #6550ad;outline-offset:3px}.actions button[hidden]{display:none}
@keyframes lens-spin{to{transform:rotate(360deg)}}@media(prefers-reduced-motion:reduce){*,*::before{animation:none!important;transition:none!important}}
@media(max-width:600px){.panel{right:12px;bottom:12px;width:264px}.badge{font-size:11px;padding:7px 9px}.badge small{font-size:9px}}
`;
// FontFace works across Chrome's shadow boundary without changing Reddit's font.
let fontLoading:Promise<unknown>|undefined;
function loadFont() {
  if(fontLoading||typeof FontFace==='undefined')return;
  const url=typeof chrome!=='undefined'&&chrome.runtime?.getURL
    ?chrome.runtime.getURL('fonts/google-sans-latin.woff2'):new URL('/fonts/google-sans-latin.woff2',location.href).href;
  const face=new FontFace('Reddit Lens Google Sans',`url("${url}")`,{weight:'400 700',style:'normal',display:'swap'});
  document.fonts.add(face);fontLoading=face.load().catch(()=>undefined);
}
function element<K extends keyof HTMLElementTagNameMap>(tag:K,className?:string) {const value=document.createElement(tag);if(className)value.className=className;return value;}

// Clip to scroll/overflow ancestors so a gallery's hidden frames never get badges.
function clippedRect(image:HTMLImageElement,topInset:number) {
  const imageStyle=getComputedStyle(image);
  if(imageStyle.visibility==='hidden'||Number(imageStyle.opacity)===0)return {left:0,top:0,width:0,height:0};
  const rect=image.getBoundingClientRect();let left=Math.max(0,rect.left),top=Math.max(topInset,rect.top),right=Math.min(innerWidth,rect.right),bottom=Math.min(innerHeight,rect.bottom);
  let parent:Element|null=image.parentElement;
  if(!parent){const root=image.getRootNode();parent=root instanceof ShadowRoot?root.host:null;}
  while(parent){
    const style=getComputedStyle(parent),clip=parent.getBoundingClientRect();
    if(Number(style.opacity)===0)return {left:0,top:0,width:0,height:0};
    if(clip.width&&clip.height){
      if(/hidden|clip|scroll|auto/.test(style.overflowX)){left=Math.max(left,clip.left);right=Math.min(right,clip.right);}
      if(/hidden|clip|scroll|auto/.test(style.overflowY)){top=Math.max(top,clip.top);bottom=Math.min(bottom,clip.bottom);}
    }
    const root=parent.getRootNode();parent=parent.parentElement??(root instanceof ShadowRoot?root.host:null);
  }
  return {left,top,width:right-left,height:bottom-top};
}
export class RedditOverlay {
  readonly host=element('div');readonly shadow:ShadowRoot;
  private layer=element('div','layer');private panel=element('section','panel');
  private state=element('span','state');private detail=element('p','detail');private counts=element('div','counts');private model=element('div','model');
  private primary=element('button','primary');private retry=element('button');
  private frames=new Map<HTMLImageElement,{node:HTMLDivElement;label:HTMLElement;score:HTMLSpanElement;detail:HTMLElement}>();
  private visible=new Set<HTMLImageElement>();private scheduled=false;
  constructor(actions:{toggle:()=>void;retry:()=>void;settings:()=>void}) {
    loadFont();this.host.id=OVERLAY_ID;this.shadow=this.host.attachShadow({mode:'open'});const style=element('style');style.textContent=CSS;this.shadow.append(style,this.layer);this.layer.append(this.panel);
    const heading=element('div','heading'),dot=element('span','dot'),brand=element('strong');brand.textContent='Reddit Lens';heading.append(dot,brand,this.state);
    this.panel.setAttribute('aria-label','Reddit Lens scanner status');this.state.setAttribute('role','status');this.state.setAttribute('aria-live','polite');
    const buttons=element('div','actions'),settings=element('button');settings.textContent='Settings';settings.onclick=actions.settings;this.primary.onclick=actions.toggle;
    this.retry.textContent='Retry';this.retry.onclick=actions.retry;buttons.append(this.primary,this.retry,settings);
    this.panel.append(heading,this.model,this.detail,this.counts,buttons);document.documentElement.append(this.host);
    addEventListener('scroll',()=>this.schedule(),{passive:true,capture:true});addEventListener('resize',()=>this.schedule(),{passive:true});
  }
  update(status:PageStatus) {
    this.panel.dataset.mode=status.mode;if(this.state.textContent!==status.label)this.state.textContent=status.label;this.detail.textContent=status.detail;
    this.counts.replaceChildren(...[['scanned',status.scanned],['AI signals',status.ai],...(status.errors?[['failed',status.errors]]:[])].map(([label,count])=>{const span=element('span'),b=element('b');b.textContent=String(count);span.append(b,document.createTextNode(` ${label}`));return span;}));
    const modelLabel=element('small'),modelName=element('b');modelLabel.textContent='Model';modelName.textContent=status.model.replace(' · experimental','');this.model.replaceChildren(modelLabel,modelName);
    this.primary.textContent=status.mode==='paused'||status.mode==='starting'?'Start scanning':status.mode==='scope'?'Scan this feed':'Pause';
    this.retry.hidden=status.mode!=='error'&&!status.errors;this.panel.title=status.model;
  }
  mark(image:HTMLImageElement,mark:ImageMark) {
    let frame=this.frames.get(image);
    if(!frame){const node=element('div','frame'),badge=element('div','badge'),head=element('div','badge-head'),spinner=element('span','spinner'),label=element('b'),score=element('span','score'),detail=element('small');
      head.append(spinner,label,score);badge.append(head,detail);node.append(badge);node.setAttribute('role','note');this.layer.prepend(node);frame={node,label,score,detail};this.frames.set(image,frame);
    }
    frame.node.dataset.state=mark.state;frame.node.dataset.busy=String(!!mark.busy);frame.node.title=mark.title;
    frame.label.textContent=mark.label;frame.score.textContent=mark.score??'';frame.detail.textContent=mark.detail;this.schedule();
  }
  visibility(image:HTMLImageElement,visible:boolean) {if(visible)this.visible.add(image);else this.visible.delete(image);this.schedule();}
  remove(image:HTMLImageElement) {this.frames.get(image)?.node.remove();this.frames.delete(image);}
  clear() {for(const frame of this.frames.values())frame.node.remove();this.frames.clear();}
  schedule() {if(this.scheduled)return;this.scheduled=true;requestAnimationFrame(()=>{this.scheduled=false;this.position();});}
  private position() {
    const drawn=new Set<string>();
    const active=[];
    for(const [image,frame] of this.frames){
      if(!image.isConnected){this.remove(image);this.visible.delete(image);continue;}
      if(document.hidden||!this.visible.has(image)){frame.node.hidden=true;continue;}
      active.push({image,frame});
    }
    // Do not force layout/style reads for every previously scanned feed image.
    if(!active.length)return;
    // Keep marks below a visible Reddit navigation bar while images scroll under it.
    const topInset=Math.max(0,...Array.from(document.querySelectorAll('header,[role="banner"],#reddit-header,shreddit-header')).map(node=>node.getBoundingClientRect()).filter(rect=>rect.top<=1&&rect.bottom>0&&rect.height<180&&rect.width>innerWidth*.6).map(rect=>rect.bottom));
    for(const {image,frame} of active){
      const rect=clippedRect(image,topInset);const key=`${image.currentSrc||image.src}:${Math.round(rect.left)}:${Math.round(rect.top)}:${Math.round(rect.width)}:${Math.round(rect.height)}`;
      const visible=this.visible.has(image)&&!document.hidden&&rect.width>=45&&rect.height>=45&&!drawn.has(key);
      frame.node.hidden=!visible;if(!visible)continue;
      drawn.add(key);
      Object.assign(frame.node.style,{left:`${rect.left}px`,top:`${rect.top}px`,width:`${rect.width}px`,height:`${rect.height}px`});
    }
  }
}
