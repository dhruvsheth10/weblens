"""Immutable, numerical MIRROR profiles accepted by the local benchmark runner."""
import hashlib
import json
CHECKPOINT_SHA='829ef07b52caa26c1cb3917e603a275cfd1c7f494f7b750f86fb218710ac66d5'
COMMUNITY_SHA='d23790ed63e8f330c7cf2770800ac0c65f81b627cced81a7a3bd3248ff532388'
UPSTREAM='adb928ec56c4ffd77c53ccb908bea29bb559922c'

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def profile(blend=False):
    value={'schemaVersion':1,'kind':'mirror-hybrid' if blend else 'mirror','model':'mirror-community-v1' if blend else 'mirror-h-v1',
        'checkpointSha256':CHECKPOINT_SHA,'upstreamCommit':UPSTREAM,'threshold':.5,
        'pipeline':'mirror-h-fp32-eager-shortmax512-crop224-pngjpeg96-or-community-v1' if blend else 'mirror-h-fp32-eager-shortmax512-crop224-pngjpeg96-v1'}
    if blend:value.update(mirrorHash=profile()['modelHash'],communityHash=COMMUNITY_SHA,componentThresholds={'mirror':.5,'community':.5})
    value['modelHash']=digest(value);return value

def validate_profile(value):
    if not isinstance(value,dict) or value.get('kind') not in {'mirror','mirror-hybrid'} or value!=profile(value.get('kind')=='mirror-hybrid'):
        raise ValueError('MIRROR profile differs from the pinned checkpoint, preprocessing or frozen decision policy.')
    return value
