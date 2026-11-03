"""Dependency-free identity and numeric validation for semantic-head artifacts."""
import hashlib
import json
import math
import re

CHECKPOINT_SHA = 'a6cc439d5a6d2dfadd60c77d27a2838ad55b34e601ecd30f46ad97266d6ac4e0'
MODEL_ID = 'community-preview-v1'
PIPELINE = 'community-vit-s224-dev-augmented-linear-v1'

def canonical_hash(profile):
    return hashlib.sha256(json.dumps({k:v for k,v in profile.items() if k!='modelHash'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_profile(profile):
    if not isinstance(profile,dict) or profile.get('schemaVersion')!=1 or profile.get('kind') not in {'semantic-head','semantic-hybrid'} or profile.get('checkpointSha256')!=CHECKPOINT_SHA:
        raise ValueError('Unsupported semantic profile or checkpoint.')
    if profile.get('modelHash')!=canonical_hash(profile):raise ValueError('Semantic profile checksum mismatch.')
    head=profile.get('head',{})
    if not isinstance(head,dict):raise ValueError('Invalid semantic head.')
    weights=head.get('weight')
    def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    if not isinstance(weights,list) or len(weights)!=384 or not all(map(finite,weights)) or not finite(head.get('bias')):
        raise ValueError('Invalid semantic head coefficients.')
    if not finite(profile.get('threshold')) or profile['threshold']!=.5 or not finite(profile.get('uncertainThreshold')) or not 0<=profile['uncertainThreshold']<=.5:
        raise ValueError('Invalid semantic decision thresholds.')
    name=profile.get('model')
    expected_pipeline=PIPELINE if profile['kind']=='semantic-head' else 'community-vit-s224-ferret-centered-or-v1'
    if profile.get('pipeline')!=expected_pipeline or not isinstance(name,str) or not re.fullmatch(r'community-[a-z0-9][a-z0-9._-]{0,65}',name):
        raise ValueError('Unsupported semantic model identity.')
    if profile['kind']=='semantic-hybrid':
        cuts=profile.get('componentThresholds',{})
        if not isinstance(cuts,dict) or cuts.get('ferret')!=0.8304235935211183 or cuts.get('semantic')!=.5 or not re.fullmatch(r'[a-f0-9]{64}',str(profile.get('semanticProfileHash',''))):
            raise ValueError('Unsupported semantic fusion rule.')
    return profile
