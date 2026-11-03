// Run with agent-browser eval --stdin in the local development lab.
// Synthetic DOM performance regression only; no images are sent to a detector.
(async () => {
  const {RedditOverlay} = await import('/src/extension/overlay.ts');
  const container=document.createElement('div');document.body.append(container);
  const overlay=new RedditOverlay({toggle(){},retry(){},settings(){}});
  const reads=Array(1000).fill(0);
  try {
    for(let i=0;i<reads.length;i++) {
      const img=document.createElement('img');container.append(img);
      img.getBoundingClientRect=()=>{reads[i]++;return new DOMRect(10,100,200,200);};
      overlay.mark(img,{state:'unmarked',label:'DOM test only',detail:'No model inference',title:'Test'});
      overlay.visibility(img,i===0);
    }
    await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
    const visibleReads=reads[0],offscreenReads=reads.slice(1).reduce((a,b)=>a+b,0);
    if(visibleReads<1 || offscreenReads!==0)throw new Error(`Unexpected layout work: ${visibleReads} visible, ${offscreenReads} offscreen`);
    return {kind:'overlay-layout-regression',trackedImages:reads.length,visibleImages:1,visibleReads,offscreenReads,passed:true};
  } finally {overlay.clear();overlay.host.remove();container.remove();}
})();
