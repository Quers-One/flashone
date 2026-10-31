"""make_icon.py - turn DSH 里的那张棋盘图 into a multi-size Windows .ico

Usage (in cmd.exe / PowerShell, as a COMMAND - not pasted into the shell):

    python make_icon.py                      # uses icon_src.png next to this file
    python make_icon.py D:\some\other.png    # or point at any PNG

Output: flash.ico next to this script, containing 16/32/48/64/128/256 px frames.
"""
import os
import sys

try:
    from PIL import Image
except ImportError:
    print("[x] Pillow missing. Install it with:  python -m pip install pillow")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "icon_src.png")
DST = os.path.join(HERE, "flash.ico")
SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

if not os.path.exists(SRC):
    print("[x] source not found:", SRC)
    print("    save the board image as PNG there first (do not use a .jpg).")
    sys.exit(1)

img = Image.open(SRC).convert("RGBA")
print("[i] source:", SRC, img.size)

# pure white -> transparent; threshold 247 so the soft board edge survives.
# raise to 250 if the plate edge gets eaten, lower to 240 if a white halo stays.
r, g, b, a = img.split()
mask = Image.merge("RGB", (r, g, b)).point(
    lambda v: 0 if v >= 247 else 255
).convert("L")
img.putalpha(Image.composite(a, Image.new("L", img.size, 0), mask))

img.save(DST, format="ICO", sizes=SIZES)
print("[+] written:", DST, str(round(os.path.getsize(DST) / 1024, 1)) + " KB")
print("[>] now: right-click the Flash UI shortcut -> Properties ->")
print("    Change Icon... -> Browse ->", DST)
