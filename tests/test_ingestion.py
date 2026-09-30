import pandas as pd
import pytest

import column_map as cm
import ingestion
from conftest import make_rows

HEADERS = ["Docket_ID", "Farmer_Name", "Crop_Name", "Latitude", "Longitude", "Surveyor_Name", "State", "District", "Weird Extra"]


def test_exact_alias_and_normalisation():
    m = cm.map_columns(["DOCKET ID", "farmer-name", "Affected_Area_%", "Crop Loss %", "lat", "Lon"])
    assert m == {"docket_id": 0, "farmer_name": 1, "affected_area_pct": 2, "crop_loss_pct": 3, "latitude": 4, "longitude": 5}


def test_each_source_column_claimed_once():
    m = cm.map_columns(["Docket_ID", "Docket_ID"])
    assert list(m.values()) == [0]


def test_fuzzy_match_typo_and_no_false_positive():
    m = cm.map_columns(["Surveyer_Name", "Latitud", "Level6_Name", "Level7_Name"])
    assert m["surveyor_name"] == 0 and m["latitude"] == 1
    # near-identical Level6/Level7 must not be confused
    assert m["village"] == 2 and m["patwar_circle"] == 3
    assert cm.detect_column(["totally unrelated", "zzz"], "docket_id") is None


def test_exact_beats_fuzzy():
    # 'latitude' exact must be taken by latitude even though 'Latitud' is fuzzy-close
    m = cm.map_columns(["Latitud", "latitude"])
    assert m["latitude"] == 1


def test_has_header():
    assert ingestion.has_header(HEADERS)
    assert not ingestion.has_header(make_rows(1)[0])
    assert not ingestion.has_header(["PM-2026-001", "APP1", "12.5", "x"])  # alphanumeric docket, numeric cell
    assert not ingestion.has_header(["https://x.invalid/a.pdf", "b"])
    assert not ingestion.has_header([None, None])


def test_headerless_xlsx(headerless_xlsx):
    df = ingestion.load(headerless_xlsx)
    assert len(df) == 6
    assert df.loc[0, "docket_id"] == "2000000000"
    assert df["latitude"].dtype.kind == "f" and df.loc[1, "latitude"] == pytest.approx(29.01)
    assert df.loc[0, "surveyor_name"] == "Surveyor 0"
    assert df["surveyor_remark"].isna().all()


def test_headered_xlsx_and_unmapped_kept(tmp_path):
    p = tmp_path / "h.xlsx"
    pd.DataFrame([["1", "F", "Paddy", "29.1", "76.1", "S", "StateA", "D", "x"]], columns=HEADERS).to_excel(p, index=False)
    df = ingestion.load(str(p))
    assert len(df) == 1 and df.loc[0, "docket_id"] == "1"
    assert df.loc[0, "latitude"] == pytest.approx(29.1)
    assert df.loc[0, "Weird Extra"] == "x"
    assert "pdf_url" in df  # missing canonical columns are added


def test_csv_and_bad_numbers(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("docketID,farmerName,latitude,longitude,Affected_Area_%\n 12.0 ,A, 29.5,oops,50\n,,,,\n")
    df = ingestion.load(str(p))
    assert len(df) == 1                       # all-blank row dropped
    assert df.loc[0, "docket_id"] == "12"     # trailing .0 and whitespace stripped
    assert pd.isna(df.loc[0, "longitude"]) and df.loc[0, "affected_area_pct"] == 50


def test_headerless_csv(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame(make_rows(3)).to_csv(p, header=False, index=False)
    df = ingestion.load(str(p))
    assert len(df) == 3 and df.loc[2, "crop_name"] == "Paddy"


def test_folder_mixed(tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    pd.DataFrame(make_rows(2)).to_excel(d / "a.xlsx", header=False, index=False)
    pd.DataFrame([["9", "F", "Paddy", "29.1", "76.1", "S", "StateA", "D", "x"]], columns=HEADERS).to_csv(d / "b.csv", index=False)
    (d / "~$lock.xlsx").write_text("junk")
    (d / "notes.txt").write_text("junk")
    df = ingestion.load(str(d))
    assert len(df) == 3 and set(df["docket_id"]) == {"2000000000", "2000000001", "9"}


def test_empty_folder_and_empty_file(tmp_path):
    (tmp_path / "e").mkdir()
    with pytest.raises(FileNotFoundError):
        ingestion.load(str(tmp_path / "e"))
    p = tmp_path / "empty.csv"
    p.write_text("docketID,farmerName,latitude\n")
    assert len(ingestion.load(str(p))) == 0
