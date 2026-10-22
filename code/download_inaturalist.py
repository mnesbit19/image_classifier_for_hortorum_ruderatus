"""We turn an iNaturalist CSV export into licensed photos and provenance records.

Python 3.9+; standard library only. Network access is to public iNaturalist/S3
endpoints. This program neither connects to HPC nor submits jobs.
"""

import argparse
import collections
import contextlib
import csv
import datetime as dt
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.inaturalist.org/v1"
VERSION = 1
BATCH_SIZE = 100
MEDIA_HOSTS = {"inaturalist-open-data.s3.amazonaws.com", "static.inaturalist.org"}
MAX_PHOTO_BYTES = 25 * 1024 * 1024
DAILY_BYTES = 12 * 1024**3


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def save_once(path, raw):
    """We publish complete files, retaining any existing, different content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("Existing immutable file differs: " + str(path))
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Our project-wide lock excludes other downloader writers. Image paths
        # also contain their content hash, so different bytes get different paths.
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_export(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError("Export ZIP must contain exactly one CSV.")
            if archive.getinfo(names[0]).file_size > 100 * 1024**2:
                raise ValueError("Export is too large for this targeted downloader.")
            text = archive.read(names[0]).decode("utf-8-sig")
    else:
        text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    required = {"id", "scientific_name", "taxon_id", "quality_grade"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("CSV is missing required observation columns: " + str(sorted(required)))
    rows = list(reader)
    ids = [r["id"] for r in rows]
    if not rows or any(not re.fullmatch(r"[1-9][0-9]*", i) for i in ids):
        raise ValueError("Export must have valid positive observation IDs.")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate observation IDs in export; resolve before retrieval.")
    if len(rows) > 10000:
        raise ValueError("More than 10,000 observations: design a bulk-data workflow first.")
    return rows


def audit(rows):
    return {
        "observations": len(rows),
        "scientific_names": dict(collections.Counter(r["scientific_name"] for r in rows)),
        "quality_grades": dict(collections.Counter(r["quality_grade"] for r in rows)),
        "countries": dict(collections.Counter(r.get("place_country_name", "") for r in rows)),
        "primary_image_links": sum(bool(r.get("image_url")) for r in rows),
        "photo_licenses_in_export": False,
        "note": "CSV license is the observation license; image_url is only the first photo.",
    }


class BudgetReached(Exception):
    pass


class HTTPFailure(RuntimeError):
    """We retain the server's explanation alongside the request that failed."""

    def __init__(self, url, status, reason, response_body):
        self.details = {"url": url, "status": status, "reason": str(reason),
                        "response_body_excerpt": response_body}
        super().__init__(f"HTTP {status} {reason}; URL={url}; response={response_body}")


class Client:
    def __init__(self, root):
        self.last_request = 0.0
        self.ledger = root / "logs/downloads/media_bytes.jsonl"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.media_events = []
        if self.ledger.exists():
            self.media_events = [json.loads(line) for line in self.ledger.read_text().splitlines()]

    def fetch(self, url, media=False):
        parsed = urllib.parse.urlsplit(url)
        hosts = MEDIA_HOSTS if media else {"api.inaturalist.org"}
        if parsed.scheme != "https" or parsed.hostname not in hosts:
            raise ValueError("Unexpected download endpoint: " + url)
        for attempt in range(4):
            time.sleep(max(0, self.last_request + 1.1 - time.monotonic()))
            if media:
                used = sum(e["bytes"] for e in self.media_events if e["time"] > time.time() - 86400)
                if used + MAX_PHOTO_BYTES > DAILY_BYTES:
                    raise BudgetReached("Project media budget reached; resume after 24 hours.")
            self.last_request = time.monotonic()
            request = urllib.request.Request(url, headers={
                "User-Agent": "ml_rudvhort/0.3 (Miles Nesbit; noncommercial bee research)",
                "Accept": "image/*" if media else "application/json",
            })
            received = 0
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    final_url = response.geturl()
                    final = urllib.parse.urlsplit(final_url)
                    if final.scheme != "https" or final.hostname not in hosts:
                        raise ValueError("Unexpected redirect endpoint: " + final_url)
                    limit = MAX_PHOTO_BYTES if media else 64 * 1024**2
                    length = response.headers.get("Content-Length")
                    if length and int(length) > limit:
                        raise ValueError("Response exceeds our per-file size limit.")
                    chunks = []
                    started = time.monotonic()
                    while True:
                        block = response.read(256 * 1024)
                        if not block:
                            break
                        chunks.append(block)
                        received += len(block)
                        if received > limit:
                            raise ValueError("Response exceeds our per-file size limit.")
                        if media:
                            # We keep media traffic below about 1 MiB/second.
                            time.sleep(max(0, received / 1024**2 - (time.monotonic() - started)))
                    if length and int(length) != received:
                        raise OSError("Truncated response (Content-Length mismatch).")
                    return b"".join(chunks), response.headers.get("Content-Type", ""), final_url
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                    body = error.read(8192).decode("utf-8", errors="replace")
                    error.close()
                    raise HTTPFailure(url, error.code, error.reason, body) from error
                retry = error.headers.get("Retry-After", "")
                error.close()
                if retry.isdigit():
                    delay = int(retry)
                else:
                    delay = 5 * 2**attempt
                if delay > 300:
                    raise BudgetReached("Server requests a long pause; resume later.") from error
                time.sleep(max(5 * 2**attempt, delay))
            except (urllib.error.URLError, OSError, socket.timeout, http.client.IncompleteRead):
                if attempt == 3:
                    raise
                time.sleep(5 * 2**attempt)
            finally:
                if media and received:
                    event = {"time": time.time(), "bytes": received}
                    self.media_events.append(event)
                    with self.ledger.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(event) + "\n")
        raise RuntimeError("Request retries exhausted.")


def species_from_taxon(taxon, taxa):
    lineage = set(taxon.get("ancestor_ids") or []) | {taxon.get("id")}
    matches = [name for name, taxon_id in taxa.items() if taxon_id in lineage]
    return matches[0] if len(matches) == 1 else None


def export_species(row, taxa):
    name = row["scientific_name"].strip()
    return next((s for s in taxa if name == s or name.startswith(s + " ")), None)


def photo_url(photo, size):
    url = urllib.parse.urlsplit(photo.get("url") or "")
    if url.scheme != "https" or url.hostname not in MEDIA_HOSTS:
        raise ValueError("Unsupported photo host or missing URL")
    # Some older photos really have an empty extension. We preserve the exact
    # suffix, then choose the local extension from the downloaded file signature.
    pattern = rf"^/photos/{int(photo['id'])}/(?:square|thumb|small|medium|large|original)(\.[A-Za-z0-9]*)$"
    match = re.fullmatch(pattern, url.path)
    if not match:
        raise ValueError("Unrecognised photo URL layout")
    return urllib.parse.urlunsplit((url.scheme, url.netloc,
        f"/photos/{photo['id']}/{size}{match.group(1)}", url.query, ""))


def image_extension(raw, content_type):
    # These checks detect HTML/error bodies and obvious truncation. Full image
    # decoding and biological quality checks are a separate curation stage.
    mime = content_type.lower().split(";")[0].strip()
    if mime and not (mime.startswith("image/") or mime == "application/octet-stream"):
        raise ValueError("Server did not return an image: " + content_type)
    if raw.startswith(b"\xff\xd8\xff") and raw.rstrip().endswith(b"\xff\xd9"):
        return ".jpg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and raw[-12:] == b"\x00\x00\x00\x00IEND\xaeB`\x82":
        return ".png"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        if int.from_bytes(raw[4:8], "little") + 8 == len(raw):
            return ".webp"
    if raw[:6] in {b"GIF87a", b"GIF89a"} and raw.endswith(b";"):
        return ".gif"
    raise ValueError("Unsupported or truncated image signature; retained as a failed download.")


def base_record(row, observation, snapshot, config):
    taxon = observation.get("taxon") or {}
    observer = observation.get("user") or {}
    species = species_from_taxon(taxon, config["data"]["taxon_ids"])
    return {
        "observation_id": int(row["id"]), "observation_url": f"https://www.inaturalist.org/observations/{row['id']}",
        "export_scientific_name": row["scientific_name"], "export_taxon_id": row["taxon_id"],
        "export_quality_grade": row["quality_grade"], "export_observation_license": row.get("license", ""),
        "current_scientific_name": taxon.get("name"), "current_taxon_id": taxon.get("id"),
        "species": species, "class_id": config["classes"].get(species),
        "quality_grade": observation.get("quality_grade"),
        "taxon_changed_since_export": str(taxon.get("id")) != row["taxon_id"],
        "observer_id": observer.get("id"), "observer_login": observer.get("login"),
        "observed_on": observation.get("observed_on"), "observation_updated_at": observation.get("updated_at"),
        "export_country": row.get("place_country_name"),
        "public_location": observation.get("location"),
        "public_positional_accuracy": observation.get("public_positional_accuracy"),
        "geoprivacy": observation.get("geoprivacy"), "taxon_geoprivacy": observation.get("taxon_geoprivacy"),
        "obscured": observation.get("obscured"),
        "identification_agreements": observation.get("num_identification_agreements"),
        "identification_disagreements": observation.get("num_identification_disagreements"),
        "metadata_snapshot": snapshot,
    }


def eligible_records(row, observation, snapshot, config, size):
    record = base_record(row, observation or {}, snapshot, config)
    reason = None
    if observation is None:
        reason = "observation_missing_from_api"
    elif record["species"] is None:
        reason = "current_taxon_outside_target_pair"
    elif export_species(row, config["data"]["taxon_ids"]) != record["species"]:
        reason = "species_changed_since_export_review_required"
    elif observation.get("quality_grade") != config["data"]["initial_quality_grade"]:
        reason = "current_quality_grade_not_research"
    elif observation.get("captive"):
        reason = "captive_observation"
    photos = (observation or {}).get("photos") or []
    if reason or not photos:
        return [], [{**record, "reason": reason or "no_photos"}]
    accepted, rejected = [], []
    for index, photo in enumerate(photos):
        license_code = (photo.get("license_code") or "").lower()
        item = {**record, "photo_id": photo.get("id"), "view_index": index,
            "photo_page": f"https://www.inaturalist.org/photos/{photo.get('id')}",
            "photo_license": license_code, "attribution": photo.get("attribution"),
            "license_evidence_url": f"https://www.inaturalist.org/photos/{photo.get('id')}",
            "license_version": None,
            "original_dimensions": photo.get("original_dimensions"),
            "api_photo_url": photo.get("url"), "requested_size": size}
        reason = None
        if license_code not in config["data"]["initial_photo_license_allowlist"]:
            reason = "photo_license_not_in_allowlist"
        elif photo.get("hidden") or photo.get("flags") or photo.get("moderator_actions"):
            reason = "hidden_or_flagged_photo_review_required"
        elif not photo.get("attribution"):
            reason = "missing_photo_attribution"
        else:
            try:
                item["download_url"] = photo_url(photo, size)
            except (ValueError, KeyError, TypeError) as error:
                reason = str(error)
        if reason:
            rejected.append({**item, "reason": reason})
        else:
            accepted.append(item)
    return accepted, rejected


def download_photo(item, root, cache, client):
    receipt = cache / "photo_receipts" / f"{item['photo_id']}.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        path = root / saved["file_path"]
        if saved["download_url"] != item["download_url"]:
            raise ValueError("Cached photo URL differs; start a new dataset snapshot.")
        if not path.is_file() or file_sha(path) != saved["sha256"]:
            raise ValueError("Cached image missing or changed: " + str(path))
        return {**item, **saved, "status": "cached"}
    raw, content_type, resolved_url = client.fetch(item["download_url"], media=True)
    extension = image_extension(raw, content_type)
    digest = sha(raw)
    relative = Path("data/raw/images") / str(item["photo_id"]) / item["requested_size"] / (digest + extension)
    save_once(root / relative, raw)
    saved = {"file_path": relative.as_posix(), "sha256": digest, "bytes": len(raw),
        "content_type": content_type, "download_url": item["download_url"],
        "resolved_url": resolved_url, "downloaded_utc": utc(),
        "validation": "signature_length_sha256_only; full_decode_and_quality_pending"}
    save_once(receipt, json_bytes(saved))
    return {**item, **saved, "status": "downloaded"}


def write_csv(path, records):
    if not records:
        save_once(path, b"observation_id,photo_id,status,reason\n")
        return
    columns = list(dict.fromkeys(k for r in records for k in r))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in record.items()})
    save_once(path, stream.getvalue().encode("utf-8"))


@contextlib.contextmanager
def project_lock(root):
    path = root / "logs/downloads/ACTIVE.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                                     "started_utc": utc(), "job_id": os.getenv("PBS_JOBID")}))
    except FileExistsError:
        raise ValueError("Downloader lock exists: " + str(path) +
                         ". Check the recorded job/process before removing a stale lock.")
    try:
        yield
    finally:
        path.unlink()


def run(args, client=None):
    root = args.root.resolve()
    source = args.export.resolve()
    config_path = root / "config/project.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config["data"].get("taxon_ids"):
        raise ValueError("Resolve and configure both taxon IDs first.")
    rows = read_export(source)
    export_audit = audit(rows)
    if args.audit_only:
        print(json.dumps(export_audit, indent=2))
        return 0
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.dataset):
        raise ValueError("Use a short dataset name containing only letters, numbers, _ or -.")
    if args.limit_observations:
        rows = rows[:args.limit_observations]
    cache = root / "data/raw/metadata" / args.dataset
    plan = {"schema_version": VERSION, "export_sha256": file_sha(source),
        "size": args.size, "observation_ids": [int(r["id"]) for r in rows],
        "data_policy": config["data"], "class_mapping": config["classes"], "batch_size": BATCH_SIZE}
    with project_lock(root):
        save_once(cache / "plan.json", json_bytes(plan))
        save_once(cache / ("source_export" + (".csv.zip" if zipfile.is_zipfile(source) else ".csv")), source.read_bytes())
        save_once(cache / "export_audit.json", json_bytes(export_audit))
        attempt = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        output = root / "data/manifests" / args.dataset / attempt
        output.mkdir(parents=True, exist_ok=False)
        save_once(output / "code_snapshot.py", Path(__file__).read_bytes())
        save_once(output / "config.json", config_path.read_bytes())
        client = client or Client(root)
        records, excluded = [], []
        fetched = new = 0
        failure = None
        http_error = None
        paused = False
        try:
            for start in range(0, len(rows), BATCH_SIZE):
                batch_rows = rows[start:start + BATCH_SIZE]
                ids = [int(r["id"]) for r in batch_rows]
                batch_path = cache / "api" / f"observations_{start:06d}.json"
                if not batch_path.exists():
                    # The ID endpoint defaults to 30 records and rejects a longer
                    # ID list unless we explicitly request a sufficient page size.
                    url = API + "/observations/" + ",".join(map(str, ids)) + f"?per_page={len(ids)}"
                    raw, _, _ = client.fetch(url)
                    response = json.loads(raw)
                    if not isinstance(response.get("results"), list):
                        raise ValueError("Unexpected API response; results list missing.")
                    save_once(batch_path, json_bytes({"fetched_utc": utc(), "url": url,
                        "requested_ids": ids, "response": response}))
                envelope = json.loads(batch_path.read_text(encoding="utf-8"))
                if envelope["requested_ids"] != ids:
                    raise ValueError("Cached observation batch does not match the export.")
                observations = {int(o["id"]): o for o in envelope["response"]["results"]}
                # A paginated/truncated result must not masquerade as deleted records.
                if envelope["response"].get("total_results", len(observations)) > len(observations):
                    raise ValueError("API batch was truncated; reduce batch size in a new snapshot.")
                for row in batch_rows:
                    items, rejects = eligible_records(row, observations.get(int(row["id"])),
                        batch_path.relative_to(root).as_posix(), config, args.size)
                    excluded.extend(rejects)
                    fetched += 1
                    for item in items:
                        if args.metadata_only:
                            records.append({**item, "status": "not_downloaded_metadata_only"})
                        elif args.max_downloads is not None and new >= args.max_downloads:
                            records.append({**item, "status": "pending_download_limit"})
                            paused = True
                        else:
                            try:
                                downloaded = download_photo(item, root, cache, client)
                                records.append(downloaded)
                                new += downloaded["status"] == "downloaded"
                            except BudgetReached:
                                raise
                            except (ValueError, OSError, urllib.error.URLError, http.client.IncompleteRead, HTTPFailure) as error:
                                records.append({**item, "status": "failed", "error": str(error)})
                    if fetched % 25 == 0 or fetched == len(rows):
                        print(f"OBSERVATIONS={fetched}/{len(rows)} NEW_PHOTOS={new} EXCLUSIONS={len(excluded)}", flush=True)
        except (Exception, KeyboardInterrupt) as error:
            failure = f"{type(error).__name__}: {error}"
            if isinstance(error, HTTPFailure):
                http_error = error.details
        counts = dict(collections.Counter(r["status"] for r in records))
        complete = not failure and fetched == len(rows) and not paused and not counts.get("failed")
        status = "METADATA_COMPLETE" if complete and args.metadata_only else "DOWNLOAD_COMPLETE" if complete else "INCOMPLETE"
        if status == "DOWNLOAD_COMPLETE" and not any(r["status"] in {"downloaded", "cached"} for r in records):
            status = "NO_ELIGIBLE_PHOTOS"
        summary = {"status": status, "dataset": args.dataset, "attempt": attempt,
            "created_utc": utc(), "host": socket.gethostname(), "pbs_job_id": os.getenv("PBS_JOBID"),
            "export_audit": export_audit, "selected_observations": len(rows),
            "processed_observations": fetched, "photo_status_counts": counts,
            "exclusion_counts": dict(collections.Counter(r["reason"] for r in excluded)),
            "eligible_species_counts": dict(collections.Counter(r["species"] for r in records)),
            "eligible_photo_license_counts": dict(collections.Counter(r["photo_license"] for r in records)),
            "error": failure, "http_error": http_error,
            "output_directory": output.relative_to(root).as_posix(),
            "plan_sha256": file_sha(cache / "plan.json"),
            "code_sha256": file_sha(Path(__file__)),
            "validation_scope": "File signatures, response lengths when supplied, SHA-256. Decoding, bee quality, labels and splits still need review."}
        write_csv(output / "photos.csv", records)
        write_csv(output / "excluded.csv", excluded)
        save_once(output / "summary.json", json_bytes(summary))
        print(json.dumps(summary, indent=2), flush=True)
        print("SUMMARY_FILE=" + str(output / "summary.json"), flush=True)
        return 0 if complete and status != "NO_ELIGIBLE_PHOTOS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--dataset", default="inat_export_v1")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--size", choices=["original", "large"], default="original")
    parser.add_argument("--audit-only", action="store_true", help="Read the CSV without networking")
    parser.add_argument("--metadata-only", action="store_true", help="Cache API records but do not fetch images")
    parser.add_argument("--limit-observations", type=int, help="Small test subset; requires a different dataset name")
    parser.add_argument("--max-downloads", type=int, help="Maximum new images this attempt; resume with the same dataset")
    args = parser.parse_args()
    if (args.limit_observations is not None and args.limit_observations < 1) or (args.max_downloads is not None and args.max_downloads < 0):
        parser.error("Limits must be positive (max-downloads may be zero).")
    try:
        return run(args)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        print("STOP: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
