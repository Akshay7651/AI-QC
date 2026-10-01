"""Fake read_form / analyse for orchestration tests (importable in spawned worker processes via --engine-module).

Behaviour is keyed on the LAST digit of the docket:
  9 -> engine raises          8 -> worker process dies the first time (marker file in $FAKE_MARK_DIR), then works
  7 -> not a Proforma-3       6 -> block officer signature missing        5 -> all photos are the form image
  4 -> worker process dies EVERY time                                     others -> clean
Env FAKE_DEATH=1 enables the worker-death behaviours (never use in-process). Env FAKE_SLEEP (seconds) slows every call (autosave / Ctrl-C tests).
"""
import os
import time

CALLS = {"form": 0, "photo": 0}


def _sleep():
    s = float(os.environ.get("FAKE_SLEEP", "0") or 0)
    if s:
        time.sleep(s)


def read_form(path, docket=None):
    CALLS["form"] += 1
    if os.environ.get("FAKE_LOG"):                    # one line per form read (kill/resume test: no finished row may be read again)
        with open(os.environ["FAKE_LOG"], "a") as f:
            f.write(f"{docket}\n")
    _sleep()
    d = str(docket)[-1:]
    if d == "9":
        raise RuntimeError("fake engine exploded")
    death = bool(os.environ.get("FAKE_DEATH"))
    if d == "4" and death:
        os._exit(3)
    if d == "8" and death:
        m = os.path.join(os.environ.get("FAKE_MARK_DIR", "."), f"died_{docket}")
        if not os.path.exists(m):
            open(m, "w").close()
            os._exit(3)
    return {"is_proforma3": d != "7", "quality": "good", "form_no": "HR0126" + str(docket)[-6:], "form_no_conf": 0.9,
            "po_id": str(docket), "po_id_matches": True, "form_area": 50.0, "form_loss": 40.0, "row_area": 50.0, "row_loss": 40.0,
            "total_row_blank": False, "farmer_signed": True, "company_signed": True, "worker_signed": True,
            "officer_signed": d != "6", "officer_stamp_only": d == "6", "overwrite_suspected": False, "confidence": 0.9,
            "field_conf": {}, "notes": []}


def analyse(paths, row, form_image=None):
    CALLS["photo"] += 1
    _sleep()
    n = len(paths)
    isf = str(row.get("docket_id"))[-1:] == "5"
    return {"photo_status": "OK", "field_photo": "standing crop", "farmer_photo": False, "photo_loss": 40, "photo_date": None,
            "photo_quality": "good", "photo_flags": [], "photo_is_form": isf, "n_form_photos": n if isf else 0, "n_duplicates": 0,
            "stamp_lat": None, "stamp_lng": None, "stamp_dist_m": 12.0, "stamp_date": "05092026", "rotated": False, "remarks": []}
