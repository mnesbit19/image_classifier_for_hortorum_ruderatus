"""We freeze photographer-disjoint splits from the authorised first-model pool."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from baseline_data import CLASSES, SPLITS, read_csv, sha256, write_csv, write_json, project_path


class UnionFind:
    def __init__(self, values):
        self.parent = {v: v for v in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def exact_ids(value):
    # We use the curation seed's semicolon delimiter for multiple memberships.
    return [v for v in value.split(";") if v]


def assign_splits(rows, seed, fractions):
    if set(fractions) != set(SPLITS) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("Invalid split fractions.")
    observations = {r["observation_id"] for r in rows}
    uf = UnionFind(observations)
    seen = {}
    observation_labels = {}
    for row in rows:
        if not all(row.get(key) for key in ("observer_id", "observation_group_id", "photo_id", "sha256")):
            raise ValueError("Missing split constraint.")
        label = int(row["class_id"])
        if CLASSES.get(row["species"]) != label or row["source_class_id"] != row["class_id"] or row["source_species"] != row["species"]:
            raise ValueError("Source label changed.")
        obs = row["observation_id"]
        if observation_labels.setdefault(obs, label) != label:
            raise ValueError("Conflicting observation labels.")
        for key in ("observer_id", "observation_group_id", "photo_id", "sha256"):
            token = (key, row[key])
            if token in seen:
                uf.union(obs, seen[token])
            seen[token] = obs
        for exact in exact_ids(row.get("exact_match_group_ids", "")):
            token = ("exact", exact)
            if token in seen:
                uf.union(obs, seen[token])
            seen[token] = obs
    components = defaultdict(list)
    for obs in sorted(observations):
        components[uf.find(obs)].append(obs)
    members = {}
    counts = {}
    for values in components.values():
        component = "splitgrp_" + hashlib.sha256("|".join(values).encode()).hexdigest()[:16]
        members[component] = values
        counts[component] = Counter(observation_labels[v] for v in values)
    totals = Counter(observation_labels.values())
    targets = {s: {c: fractions[s] * totals[c] for c in (0, 1)} for s in SPLITS}
    allocated = {s: Counter() for s in SPLITS}
    # Large and minority-class components are allocated first. The seed only
    # resolves order/ties; we do not search seeds using eventual model results.
    def order(component):
        size = max(counts[component][c] / totals[c] for c in (0, 1))
        return -size, hashlib.sha256(f"{seed}|{component}".encode()).hexdigest()
    assignments = {}
    for component in sorted(members, key=order):
        def cost(split):
            delta = sum(((allocated[split][c] + counts[component][c] - targets[split][c]) ** 2
                         - (allocated[split][c] - targets[split][c]) ** 2) / targets[split][c]
                        for c in (0, 1))
            return delta, SPLITS.index(split)
        split = min(SPLITS, key=cost)
        assignments[component] = split
        allocated[split].update(counts[component])
    observation_components = {obs: component for component, values in members.items() for obs in values}
    links = []
    for row in rows:
        updated = dict(row)
        component = observation_components[row["observation_id"]]
        updated.update(split=assignments[component], component_id=component)
        links.append(updated)
    validate_constraints(links)
    return links


def validate_constraints(links):
    seen = {}
    for row in links:
        for field in ("observer_id", "observation_id", "observation_group_id", "photo_id", "sha256", "component_id"):
            token = (field, row[field])
            if seen.setdefault(token, row["split"]) != row["split"]:
                raise ValueError("Leakage across splits: " + field)
        for exact in exact_ids(row.get("exact_match_group_ids", "")):
            token = ("exact", exact)
            if seen.setdefault(token, row["split"]) != row["split"]:
                raise ValueError("Exact-match leakage.")
    for split in SPLITS:
        if {int(r["class_id"]) for r in links if r["split"] == split} != {0, 1}:
            raise ValueError("A split lacks one species.")


def unique_photos(links):
    by_photo = defaultdict(list)
    for row in links:
        by_photo[row["photo_id"]].append(row)
    photos = []
    for photo_id, values in sorted(by_photo.items(), key=lambda v: int(v[0])):
        fields = ("file_path", "sha256", "bytes", "species", "class_id", "split", "component_id",
                  "source_dataset", "source_attempt", "source_download_status", "source_acceptance")
        for field in fields + ("observation_group_id",):
            if len({r[field] for r in values}) != 1:
                raise ValueError("Inconsistent shared photo metadata: " + photo_id + " " + field)
        photo = {field: values[0][field] for field in fields}
        photo.update(photo_id=photo_id, evaluation_group_id=values[0]["observation_group_id"],
                     observation_ids="|".join(sorted({r["observation_id"] for r in values})),
                     observer_ids="|".join(sorted({r["observer_id"] for r in values})))
        photos.append(photo)
    return photos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/splits_v1.json", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = project_path(root, config["input_manifest"])
    if sha256(source) != config["input_sha256"]:
        raise SystemExit("STOP: selected-pool hash changed.")
    rows = read_csv(source)
    if len(rows) != config["expected_links"] or len({r["observation_id"] for r in rows}) != config["expected_observations"]:
        raise SystemExit("STOP: selected-pool counts changed.")
    if any(r["prototype_pool_status"] != "selected_pending_split" or r["split"] for r in rows):
        raise SystemExit("STOP: unexpected selection status or prior split.")
    links = assign_splits(rows, config["seed"], config["fractions"])
    photos = unique_photos(links)
    if len(photos) != config["expected_photos"]:
        raise SystemExit("STOP: unique photo count changed.")
    summary = {"status": "SPLITS_FROZEN", "test_model_evaluation": "NOT_RUN", "seed": config["seed"],
               "constraints": config["constraints"], "leakage_checks": "PASS", "splits": {}}
    for split in SPLITS:
        selected = [r for r in links if r["split"] == split]
        unique = [r for r in photos if r["split"] == split]
        summary["splits"][split] = {
            "photos": len(unique), "links": len(selected),
            "observations": len({r["observation_id"] for r in selected}),
            "observers": len({r["observer_id"] for r in selected}),
            "components": len({r["component_id"] for r in selected}),
            "evaluation_groups": len({r["observation_group_id"] for r in selected}),
            "species": {species: {"photos": sum(r["species"] == species for r in unique),
                                   "observations": len({r["observation_id"] for r in selected if r["species"] == species})}
                        for species in CLASSES},
        }
    output = project_path(root, config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "photos.csv", photos)
    write_csv(output / "links.csv", links)
    write_json(output / "summary.json", summary)
    write_json(output / "split_config.json", config)
    write_json(output / "provenance.json", {
        "input": config["input_manifest"], "input_sha256": sha256(source),
        "builder_sha256": sha256(__file__), "helper_sha256": sha256(Path(__file__).with_name("baseline_data.py")),
        "classes": CLASSES, "hortorum_history": "INCOMPLETE; accepted saved budget-limited subset",
        "raw_images": "UNCHANGED", "labels": "SOURCE_COMMUNITY_LABELS_UNCHANGED",
        "test_policy": "Fixed now; no test predictions during baseline training or model selection.",
    })
    write_json(output / "OUTPUT_SHA256.json", {p.name: sha256(p) for p in sorted(output.iterdir())})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
