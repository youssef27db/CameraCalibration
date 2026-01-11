import cv2, glob, os
import numpy as np

src="/data/external/LiFCal/LiFCal_Data/Recalibration/LiFCal_Imageset/depth/Down1"
dst="/data/external/LiFCal/LiFCal_Data/Recalibration/LiFCal_Imageset/depth/Down1"
os.makedirs(dst, exist_ok=True)

paths=sorted(glob.glob(src+"/*.png"))
print("found", len(paths), "in", src)
if not paths:
    raise SystemExit("No PNGs found - check src path")

for p in paths:
    im=cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None:
        print("skip unreadable:", p)
        continue

    # 1) auf 1 channel reduzieren (falls RGB/RGBA)
    if im.ndim == 3:
        im = im[...,0]

    # 2) auf uint16 bringen
    if im.dtype == np.uint16:
        u16 = im
    elif im.dtype == np.uint8:
        u16 = (im.astype(np.uint16) * 257)  # 0..255 -> 0..65535
    else:
        f = im.astype(np.float32)
        f = np.clip(f, 0.0, 1.0)
        u16 = (f * 65535.0).astype(np.uint16)

    # 3) WICHTIG: auf 1280x720 (wie Settings.yaml rawImageWidth/Height)
    u16 = cv2.resize(u16, (1280, 720), interpolation=cv2.INTER_NEAREST)

    out=os.path.join(dst, os.path.basename(p))
    ok=cv2.imwrite(out, u16)
    if not ok:
        print("FAILED write:", out)

print("done ->", dst)

# sanity check
check=sorted(glob.glob(dst+"/*.png"))[:5]
for p in check:
    im=cv2.imread(p, cv2.IMREAD_UNCHANGED)
    print("CHECK", os.path.basename(p), im.shape, im.dtype, int(im.min()), int(im.max()))