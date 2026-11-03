# MIRROR reference source

`upstream.py` is the unmodified `models/mirror.py` from
https://github.com/handsome-rich/MIRROR at commit
`adb928ec56c4ffd77c53ccb908bea29bb559922c`.

The local adapter builds DINOv3 H+ from configuration and strictly loads every
checkpoint tensor, instead of calling the gated `from_pretrained` endpoint.
The upstream README advertises MIT, but the repository did not contain the linked LICENSE file at this revision; this
research integration does not claim a permissive redistribution license.
DINOv3 backbone terms also apply. Keep weights local pending distribution review.
