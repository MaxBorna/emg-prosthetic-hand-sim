"""Phase 1 baseline: RMS/MAV/WL features + LDA/SVM on NinaPro DB2, exercise 2.

Two evaluations:
  within   train on repetitions 1,3,4,6 and test on 2,5 of the SAME subject
  loso     train on all other subjects, test on the held-out one

Examples:
    python scripts/train_emg_baseline.py --subjects 1 2 3
    python scripts/train_emg_baseline.py --subjects 1 2 3 --classes 0 18 32 --clf svm
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from handsim.emg import load_db2, make_dataset, split_by_repetition
from handsim.emg.classify import make_classifier
from handsim.emg.loader import DB2_DIR, E2_MOVEMENTS
from handsim.eval import summarize
from handsim.paths import RESULTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="label ids to keep (default: all of E2 incl. rest)")
    ap.add_argument("--clf", default="lda", choices=["lda", "svm"])
    ap.add_argument("--root", default=str(DB2_DIR))
    args = ap.parse_args()

    data = {}
    for s in args.subjects:
        rec = load_db2(s, exercise=2, root=args.root)
        data[s] = make_dataset(rec, classes=args.classes)
        print(f"S{s}: {len(data[s][0])} windows, {len(np.unique(data[s][1]))} classes")

    results = {"within": {}, "loso": {}}
    for s, (X, y, r) in data.items():
        Xtr, ytr, Xte, yte = split_by_repetition(X, y, r)
        clf = make_classifier(args.clf).fit(Xtr, ytr)
        results["within"][s] = summarize(yte, clf.predict(Xte))

    if len(data) > 1:
        for s in data:
            Xtr = np.concatenate([data[o][0] for o in data if o != s])
            ytr = np.concatenate([data[o][1] for o in data if o != s])
            clf = make_classifier(args.clf).fit(Xtr, ytr)
            results["loso"][s] = summarize(data[s][1], clf.predict(data[s][0]))

    n_cls = len(np.unique(np.concatenate([d[1] for d in data.values()])))
    print(f"\n{args.clf.upper()}, {n_cls} classes, chance (balanced) = {1 / n_cls:.1%}")
    print(f"{'subject':>8} {'within acc':>11} {'within bal':>11} {'LOSO bal':>9}")
    for s in data:
        w = results["within"][s]
        lo = results["loso"].get(s, {}).get("balanced_accuracy", float("nan"))
        print(f"{s:>8} {w['accuracy']:>11.1%} {w['balanced_accuracy']:>11.1%} {lo:>9.1%}")
    if args.classes:
        print("classes:", ", ".join(f"{c}={E2_MOVEMENTS.get(c, '?')}" for c in args.classes))

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"{args.clf}_{'all' if not args.classes else '-'.join(map(str, args.classes))}"
    out = RESULTS_DIR / f"emg_baseline_{tag}.json"
    out.write_text(json.dumps({"args": vars(args), **results}, indent=1, default=str))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
