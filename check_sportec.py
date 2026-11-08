"""Sportec 공개 데이터(7경기 x 3종류 = XML 21개)가 제대로 있는지 확인.

  python check_sportec.py                 # PathCRF 저장소의 data/sportec 아래를 검사
  python check_sportec.py --find ~/Downloads   # 받아 둔 폴더에서 찾아서 어디로 옮길지 알려 줌

필요한 파일 (경기 ID: J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9)
  metadata/  DFL_02_01_matchinformation_..._DFL-MAT-<경기ID>.xml   (수십 KB)
  event/     DFL_03_02_events_raw_..._DFL-MAT-<경기ID>.xml         (수백 KB~수 MB)
  tracking/  DFL_04_03_positions_raw_observed_..._DFL-MAT-<경기ID>.xml (경기당 약 350~420 MB)
"""
import os, sys, argparse
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
KINDS = {"metadata": ("matchinformation", 5e3), "event": ("events", 5e4), "tracking": ("positions", 2e8)}


def head_ok(p):
    with open(p, "rb") as f:
        h = f.read(200).lstrip()
    return h.startswith(b"<?xml") or h.startswith(b"<")  and not h.lower().startswith(b"<!doctype html")


def classify(name):
    for kind, (key, _) in KINDS.items():
        if key in name and name.endswith(".xml"):
            for m in MATCHES:
                if m in name:
                    return kind, m
    return None, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--find", default=None)
    a = ap.parse_args()
    base = os.path.join(os.path.expanduser(os.environ.get("PATHCRF_DIR", "~/pathcrf")), "data", "sportec")
    if a.find:
        found = 0
        for root, _, files in os.walk(os.path.expanduser(a.find)):
            for fn in files:
                kind, m = classify(fn)
                if kind:
                    found += 1; p = os.path.join(root, fn)
                    print(f"{m} {kind:8s} {os.path.getsize(p)/1e6:8.1f} MB  {p}  ->  {os.path.join(base, kind)}/")
        print(f"찾은 파일 {found}개 / 필요한 파일 21개"); return
    bad = 0
    for kind, (key, minsize) in KINDS.items():
        d = os.path.join(base, kind)
        names = os.listdir(d) if os.path.isdir(d) else []
        for m in MATCHES:
            hit = [n for n in names if key in n and m in n and n.endswith(".xml")]
            if not hit:
                print(f"없음   {kind:8s} {m}"); bad += 1; continue
            p = os.path.join(d, hit[0]); sz = os.path.getsize(p)
            ok = sz >= minsize and head_ok(p)
            print(f"{'정상' if ok else '이상'}   {kind:8s} {m} {sz/1e6:8.1f} MB {'' if ok else '(크기가 너무 작거나 XML 이 아님: 다시 받기)'}")
            bad += (not ok)
    print("모두 정상" if bad == 0 else f"문제 {bad}개")


if __name__ == "__main__":
    main()
