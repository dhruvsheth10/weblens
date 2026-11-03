"""Offline MIRROR H+ adapter. No pretrained downloads or filename/Reddit features."""
from __future__ import annotations
import argparse
import io
import json
import os
import time
import torch
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode
from transformers import DINOv3ViTConfig, DINOv3ViTModel
from peft import LoraConfig, get_peft_model
from .assets import MODELS, sha256
from .image_io import decode_image, sigmoid
from .vendor.mirror.upstream import Norm, MirrorMemoryBank, DualBranchClassifier, MIRROR_Detector, DINO
from .mirror_profile import validate_profile


def backbone_config():
    return DINOv3ViTConfig(hidden_size=1280, intermediate_size=5120, num_hidden_layers=32,
        num_attention_heads=20, num_register_tokens=4, hidden_act='silu', use_gated_mlp=True,
        image_size=224, patch_size=16, rope_theta=100., pos_embed_rescale=2.,
        layer_norm_eps=1e-5, key_bias=False, query_bias=True, value_bias=True,
        mlp_bias=True, proj_bias=True, layerscale_value=1., attn_implementation='eager')


class LocalDINO(DINO):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.dino = DINOv3ViTModel(backbone_config())
        self.is_v3 = True
        self._apply_lora()


class CachedMemoryBank(MirrorMemoryBank):
    """Cache only frozen, image-independent projections; preserve sparse attention."""
    def __init__(self,feature_dim):
        super().__init__(feature_dim)
        self.register_buffer('_cached_k',None,persistent=False)
        self.register_buffer('_cached_v',None,persistent=False)

    def _apply(self,fn,recurse=True):
        self._cached_k=self._cached_v=None
        return super()._apply(fn,recurse)

    def forward(self,x):
        if self.training or torch.is_grad_enabled():
            return super().forward(x)
        b,n,c=x.shape
        if b!=1:return super().forward(x)
        q=self.q_proj(x).reshape(b,n,self.num_heads,self.head_dim).transpose(1,2)
        if self._cached_k is None:
            memory=self.memory.unsqueeze(0).expand(b,-1,-1)
            self._cached_k=self.k_proj(memory).reshape(b,-1,self.num_heads,self.head_dim).transpose(1,2)
            self._cached_v=self.v_proj(memory).reshape(b,-1,self.num_heads,self.head_dim).transpose(1,2)
        logits=(q@self._cached_k.transpose(-2,-1))*self.scale
        indices=torch.topk(logits,k=self.top_k,dim=-1).indices
        mask=torch.zeros_like(logits,dtype=torch.bool).scatter_(-1,indices,True)
        masked=logits.masked_fill(~mask,float('-inf'))
        if masked.dtype in (torch.float16,torch.bfloat16):masked=masked.float()
        attention=torch.softmax(masked,dim=-1).to(self._cached_v.dtype)
        reconstruction=(attention@self._cached_v).transpose(1,2).reshape(b,n,c)
        return self.out_proj(reconstruction),attention


class LocalMIRROR(MIRROR_Detector):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.norm = Norm(mode='imagenet')
        self.backbone = LocalDINO()
        self.memory_bank = CachedMemoryBank(1280)
        self.detector = DualBranchClassifier(1280)


def preprocess(image, png=False):
    """Authors' evaluation geometry; use decoded codec, never filename or label."""
    w,h=image.size
    if min(w,h)>512:
        scale=512/min(w,h)
        image=TF.resize(image,[round(h*scale),round(w*scale)],interpolation=InterpolationMode.BICUBIC,antialias=True)
    image=TF.center_crop(image,[224,224])
    if png:
        buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=96,optimize=True)
        with Image.open(io.BytesIO(buffer.getvalue())) as decoded:image=decoded.convert('RGB')
    return TF.to_tensor(image)


class MirrorEngine:
    def __init__(self,profile=None):
        started=time.perf_counter()
        self.profile=validate_profile(profile or json.loads((MODELS/'mirror-profile.json').read_text()))
        if self.profile['kind']!='mirror':raise ValueError('Expected standalone MIRROR profile.')
        path=MODELS/'mirror-src/checkpoint-h-cur.pth'
        if not path.is_file() or sha256(path)!=self.profile['checkpointSha256']:
            raise RuntimeError('MIRROR checkpoint missing or checksum mismatch: backend/models/mirror-src/checkpoint-h-cur.pth')
        requested=os.environ.get('LENS_DEVICE','auto')
        device=('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if requested=='auto' else requested
        if device not in {'cpu','mps','cuda'}:raise ValueError('Invalid LENS_DEVICE')
        self.device=torch.device(device);torch.set_num_threads(min(4,os.cpu_count() or 1))
        # Only the known Namespace type is allowlisted; no unrestricted pickle execution.
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        with torch.device('meta'):self.model=LocalMIRROR()
        state={k.replace('backbone.dino.layer.','backbone.dino.model.layer.'):v for k,v in checkpoint['model'].items()}
        self.model.load_state_dict(state,strict=True,assign=True)
        # RoPE uses a nonpersistent buffer; construct it on CPU instead of copying meta data.
        from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTRopePositionEmbedding
        self.model.backbone.dino.rope_embeddings=DINOv3ViTRopePositionEmbedding(backbone_config())
        self.model.requires_grad_(False).eval().to(self.device)
        self.parameter_count=sum(p.numel() for p in self.model.parameters())
        self.load_ms=(time.perf_counter()-started)*1000

    @torch.inference_mode()
    def predict(self,data:bytes,variant='original'):
        started=time.perf_counter();image,size=decode_image(data,variant)
        # Transformed JPEG/WebP is already lossy. Resize-only preserves source codec.
        with Image.open(io.BytesIO(data)) as source:png=source.format=='PNG' and variant in {'original','resize512'}
        pixels=preprocess(image,png).unsqueeze(0).to(self.device)
        logits=self.model(pixels)[0][0].float().cpu()
        logit=float(logits[1]-logits[0]);score=sigmoid(logit)
        return {'model':self.profile['model'],'modelHash':self.profile['modelHash'],'pipeline':self.profile['pipeline'],
            'score':score,'logit':logit,'width':size[0],'height':size[1],
            'elapsedMs':round((time.perf_counter()-started)*1000),'engine':self.device.type,'variant':variant,
            'watermarkVerified':False,'components':{'mirror':{'score':score,'logit':logit,'weight':1.,
                'crop':[224,224],'maxShortEdge':512,'pngJpeg96':png}}}


class MirrorCommunityEngine:
    def __init__(self,mirror=None,community=None,profile=None):
        from .community import CommunityFerretEngine
        self.profile=validate_profile(profile or json.loads((MODELS/'mirror-community-profile.json').read_text()))
        if self.profile['kind']!='mirror-hybrid':raise ValueError('Expected MIRROR combination profile.')
        self.mirror=mirror or MirrorEngine();self.community=community or CommunityFerretEngine();self.device=self.mirror.device
        if self.mirror.profile['modelHash']!=self.profile['mirrorHash'] or self.community.profile['modelHash']!=self.profile['communityHash']:
            raise ValueError('MIRROR blend component identity mismatch.')

    def predict(self,data:bytes,variant='original'):
        started=time.perf_counter();mirror=self.mirror.predict(data,variant);community=self.community.predict(data,variant)
        logit=max(mirror['logit'],community['logit'])
        return {**mirror,'model':self.profile['model'],'modelHash':self.profile['modelHash'],'pipeline':self.profile['pipeline'],
            'score':sigmoid(logit),'logit':logit,'elapsedMs':round((time.perf_counter()-started)*1000),
            'components':{**mirror['components'],**community['components']},
            'componentTimingsMs':{'mirror':mirror['elapsedMs'],'community':community['elapsedMs']},
            'decisionRule':'Frozen OR of MIRROR at 0.5 and Community + Ferret at 0.5; experimental score, not a probability'}
