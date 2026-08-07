from pathlib import Path

MARKERS = ("<<<<<<<", "=======", ">>>>>>>")

files = sorted(list(Path("data").rglob("*.csv")) + list(Path("models").rglob("*.csv")))
for f in files:
    lines = f.read_text(encoding="utf-8").splitlines()
    clean = [ln for ln in lines if not ln.startswith(MARKERS)]
    if len(clean) != len(lines):
        f.write_text("\n".join(clean) + "\n", encoding="utf-8")
        print(f"🧹 {f}: {len(lines) - len(clean)} baris konflik dibuang")
    else:
        print(f"✅ {f}: bersih")