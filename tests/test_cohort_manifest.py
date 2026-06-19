"""
test_cohort_manifest.py - validate analysis/build_cohort_manifest.py.

Builds a mock share (MUDI_synb0 subject dirs + Subjects/SubjectsInfo.txt) and asserts the generator
infers group from the id, joins age/sex correctly, excludes dotfiles, passes montages through, and
fails loudly on an id with no demographics or an un-inferable group.

Usage:  simnibs_python tests/test_cohort_manifest.py
"""
import os, sys, tempfile, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import importlib
bcm = importlib.import_module("build_cohort_manifest")


def make_share(tmp, info_rows, mudi_ids, name="share"):
    share = os.path.join(tmp, name)
    os.makedirs(os.path.join(share, "Subjects"))
    mudi = os.path.join(share, "MUDI_synb0"); os.makedirs(mudi)
    for sid in mudi_ids:
        os.makedirs(os.path.join(mudi, sid))
    open(os.path.join(mudi, ".DS_Store"), "w").close()      # dotfile must be ignored
    with open(os.path.join(share, "Subjects", "SubjectsInfo.txt"), "w") as f:
        for sid, age, sex in info_rows:
            f.write(f"{sid}\t{age}\t{sex}\n")
    return share


def main():
    tmp = tempfile.mkdtemp()
    checks = []
    def ck(name, ok, detail=""):
        checks.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    try:
        # group inference (synthetic ids only; real ids embed scan dates and must not enter the repo)
        ck("Patient id -> PD", bcm.infer_group("MockPatient1") == "PD")
        ck("Control id -> HC", bcm.infer_group("MockControl1") == "HC")
        try:
            bcm.infer_group("MockUnknown1"); ck("un-inferable id raises", False, "no error raised")
        except ValueError:
            ck("un-inferable id raises", True)

        # happy-path build: 2 Patient + 2 Control, plus a dotfile in MUDI
        info = [("MockPatient1", 53, "M"), ("MockPatient2", 71, "M"),
                ("MockControl1", 50, "M"), ("MockControl2", 54, "F")]
        ids = [r[0] for r in info]
        share = make_share(tmp, info, ids)
        man = bcm.build(share, ["M1", "DLPFC"])
        subs = man["subjects"]
        ck("all subjects captured", len(subs) == 4, f"got {len(subs)}")
        ck("group split 2 PD / 2 HC",
           sum(s["group"] == "PD" for s in subs) == 2 and sum(s["group"] == "HC" for s in subs) == 2)
        byid = {s["id"]: s for s in subs}
        ck("age joined", byid["MockPatient2"]["age"] == 71, str(byid["MockPatient2"]["age"]))
        ck("sex joined", byid["MockControl2"]["sex"] == "F", byid["MockControl2"]["sex"])
        ck("dotfile excluded", ".DS_Store" not in byid)
        ck("montages passed through", man["montages"] == ["M1", "DLPFC"], str(man["montages"]))
        ck("subjects sorted by id", [s["id"] for s in subs] == sorted(ids))

        # a MUDI subject with no SubjectsInfo row must fail loudly, not silently drop
        share2 = make_share(tmp, info[:3], ids, name="share2")   # info missing the 4th subject
        try:
            bcm.build(share2, ["M1"]); ck("missing demographics raises", False, "no error raised")
        except ValueError as e:
            ck("missing demographics raises", "absent from SubjectsInfo" in str(e))

        print(f"\n{'ALL PASS' if all(checks) else 'SOME CHECKS FAILED'}")
        sys.exit(0 if all(checks) else 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
