"""HMM search for final EM3DFold chain profiles against protein and nucleic-acid FASTA databases."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import em3dfold
import pyhmmer
from Bio import SeqIO
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from em3dfold.utils.misc_utils import abspath
from em3dfold.utils.log_utils import progress

HIT_FIELDNAMES = [
    "query_name",
    "query_kind",
    "query_len",
    "target_name",
    "accession",
    "evalue",
    "score",
    "bias",
    "description",
    "rank",
    "search_alphabet",
    "source_profile",
]

HMM_SEARCH_CONTACT_LINES = (
    f"Version: {getattr(em3dfold, '__version__', 'unknown')}",
    "Tao Li, Huang-lab, Huazhong University of Science and Technology (HUST)",
)


CHAIN_SUMMARY_FIELDNAMES = [
    "query_name",
    "query_kind",
    "query_len",
    "search_alphabet",
    "source_profile",
    "num_hits",
    "top_target_name",
    "top_evalue",
    "top_score",
]


def _format_wall_time(timestamp=None):
    dt = datetime.now() if timestamp is None else datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _safe_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        return "unnamed"
    return "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in text)


def _read_text_fasta(
    path: str,
    *,
    nucleic: bool,
    max_target_len: Optional[int] = None,
) -> Tuple[List[Tuple[str, str, str]], int, int]:
    records: List[Tuple[str, str, str]] = []
    filtered_count = 0
    max_kept_len = 0
    for record in SeqIO.parse(path, "fasta"):
        seq = str(record.seq).upper().replace(" ", "").replace("\n", "").replace("\r", "")
        seq = seq.replace("-", "")
        if nucleic:
            seq = seq.replace("T", "U")
        if max_target_len is not None and len(seq) > max_target_len:
            filtered_count += 1
            continue
        records.append((record.id, seq, record.description or record.id))
        if len(seq) > max_kept_len:
            max_kept_len = len(seq)
    return records, filtered_count, max_kept_len


def _digitize_records(
    records: Sequence[Tuple[str, str, str]],
    alphabet: pyhmmer.easel.Alphabet,
) -> List[pyhmmer.easel.DigitalSequence]:
    sequences: List[pyhmmer.easel.DigitalSequence] = []
    for name, sequence, description in records:
        text_seq = pyhmmer.easel.TextSequence(
            name=name.encode("utf-8"),
            sequence=sequence,
            description=description.encode("utf-8"),
        )
        sequences.append(text_seq.digitize(alphabet))
    return sequences


def _write_tsv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_fasta(path: Path, records: Sequence[Tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seq_records = [SeqRecord(Seq(seq), id=name, description=desc) for name, seq, desc in records]
    SeqIO.write(seq_records, path, "fasta")


def _sort_hit_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda r: (
            str(r.get("query_kind", "")),
            str(r.get("query_name", "")),
            float(r.get("evalue", 1e30)),
            -float(r.get("score", 0.0)),
            str(r.get("target_name", "")),
        ),
    )


def _make_chain_summary_row(
    *,
    query_name: str,
    query_kind: str,
    query_len: int,
    search_alphabet: str,
    source_profile: str,
    chain_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    top_row = chain_rows[0] if chain_rows else None
    return {
        "query_name": query_name,
        "query_kind": query_kind,
        "query_len": query_len,
        "search_alphabet": search_alphabet,
        "source_profile": source_profile,
        "num_hits": len(chain_rows),
        "top_target_name": "" if top_row is None else top_row.get("target_name", ""),
        "top_evalue": "" if top_row is None else top_row.get("evalue", ""),
        "top_score": "" if top_row is None else top_row.get("score", ""),
    }


def _announce_stage(stage_idx: int, total_stages: int, stage_name: str) -> float:
    progress("")
    progress(f"===== Stage {stage_idx}/{total_stages}: {stage_name} =====")
    return __import__("time").time()


def _finish_stage(start_time: float, *, skipped: bool = False) -> None:
    elapsed = 0.0 if skipped else (__import__("time").time() - start_time)
    suffix = " (skipped)" if skipped else ""
    progress(f"Finished in {elapsed:.2f} seconds{suffix}")


def _collect_profile_dirs(input_dir: str) -> Dict[str, Path]:
    root = Path(input_dir).expanduser().resolve()
    hmm_root = root / "hmm_profiles"
    candidates: Dict[str, Path] = {}

    top_after = hmm_root / "after_prune"
    if (top_after / "profiles.json").is_file():
        candidates["all"] = top_after
        return candidates

    for kind in ("protein", "na"):
        prof = hmm_root / kind / "after_prune"
        if (prof / "profiles.json").is_file():
            candidates[kind] = prof

    if candidates:
        return candidates

    if (hmm_root / "profiles.json").is_file():
        candidates["all"] = hmm_root
        return candidates
    if hmm_root.is_dir() and list(hmm_root.glob("*.hmm")):
        candidates["all"] = hmm_root
    return candidates


def _load_profile_entries(profile_dir: Path) -> List[Dict[str, object]]:
    summary_path = profile_dir / "profiles.json"
    if summary_path.is_file():
        with open(summary_path, "r") as handle:
            entries = json.load(handle)
        if isinstance(entries, list):
            return entries
    return [
        {
            "name": path.stem,
            "type": "na" if ".rna." in path.name.lower() or ".dna." in path.name.lower() else "protein",
            "alphabet": "RNA" if ".rna." in path.name.lower() or ".dna." in path.name.lower() else "amino",
            "length": None,
            "path": str(path),
        }
        for path in sorted(profile_dir.glob("*.hmm"))
    ]


def _load_hmms(profile_dir: Path, mol: str) -> List[Dict[str, object]]:
    entries = _load_profile_entries(profile_dir)
    selected: List[Dict[str, object]] = []
    for entry in entries:
        entry_type = str(entry.get("type", "")).lower()
        if mol == "protein" and entry_type != "protein":
            continue
        if mol == "na" and entry_type != "na":
            continue
        if mol == "all" and entry_type not in {"protein", "na"}:
            continue
        path = Path(str(entry.get("path", ""))).expanduser()
        if not path.is_file():
            alt = profile_dir / path.name
            if alt.is_file():
                path = alt
        if not path.is_file():
            continue
        hmm = pyhmmer.plan7.HMMFile(str(path)).read()
        selected.append({**entry, "path": str(path), "hmm": hmm})
    return selected


def _filter_by_length(entries: List[Dict[str, object]], min_chain_len: int) -> List[Dict[str, object]]:
    kept: List[Dict[str, object]] = []
    for entry in entries:
        length = int(entry.get("length") or 0)
        if length >= min_chain_len:
            kept.append(entry)
    return kept


def _search_one_mol(
    *,
    mol: str,
    hmm_entries: Sequence[Dict[str, object]],
    fasta_path: str,
    alphabet: pyhmmer.easel.Alphabet,
    output_dir: Path,
    evalue: float,
    F1: float,
    F2: float,
    F3: float,
    T: Optional[float],
    cpus: int,
    topk: int,
    write_a2m: bool,
    write_report: bool,
    max_target_len: Optional[int],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Tuple[str, str, str]]]:
    hmms = list(hmm_entries)
    if not hmms:
        raise RuntimeError(f"No valid {mol} HMM profiles were provided for search")

    print(f"# HMM search {mol}: loading FASTA from {fasta_path}")
    fasta_records, filtered_target_count, max_kept_target_len = _read_text_fasta(
        fasta_path,
        nucleic=(mol == "na"),
        max_target_len=max_target_len,
    )
    if not fasta_records:
        raise RuntimeError(f"No sequences were found in {fasta_path}")
    print(
        f"# HMM search {mol}: loaded {len(fasta_records)} target sequences "
        f"(filtered_long={filtered_target_count}, max_kept_len={max_kept_target_len})"
    )
    print(f"# HMM search {mol}: digitizing target sequences")
    digitized = _digitize_records(fasta_records, alphabet)
    target_map = {name: (name, seq, desc) for name, seq, desc in fasta_records}

    print(
        f"# HMM search {mol}: running hmmsearch with {len(hmms)} query chains, "
        f"{len(digitized)} target sequences, cpus={cpus}"
    )
    all_hits = pyhmmer.hmmer.hmmsearch(
        [item["hmm"] for item in hmms],
        digitized,
        F1=F1,
        F2=F2,
        F3=F3,
        E=evalue,
        T=T,
        cpus=cpus,
    )

    mol_dir = output_dir / mol
    mol_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []
    chain_summary_rows: List[Dict[str, object]] = []
    selected_names = set()
    selected_records: List[Tuple[str, str, str]] = []

    print(f"# HMM search {mol}: collecting per-chain hits")
    for hits, item in tqdm(zip(all_hits, hmms), total=len(hmms), desc=f"hmm_search[{mol}]", unit="chain"):
        query_name = str(item.get("name", Path(str(item["path"])).stem))
        query_len = int(item.get("length") or 0)
        query_kind = str(item.get("type", mol))
        search_alphabet = "rna" if mol == "na" else "amino"
        source_profile = str(item["path"])

        if write_report:
            report_path = mol_dir / f"{_safe_name(query_name)}.hmmsearch.txt"
            with open(report_path, "wb") as handle:
                hits.write(handle)
        if write_a2m:
            try:
                msa = hits.to_msa(alphabet)
                with open(mol_dir / f"{_safe_name(query_name)}.a2m", "wb") as handle:
                    msa.write(handle, "a2m")
            except Exception:
                pass

        chain_rows: List[Dict[str, object]] = []
        for hit in hits:
            if float(hit.evalue) > evalue:
                continue
            row = {
                "query_name": query_name,
                "query_kind": query_kind,
                "query_len": query_len,
                "target_name": hit.name.decode("utf-8") if isinstance(hit.name, (bytes, bytearray)) else str(hit.name),
                "accession": hit.accession.decode("utf-8") if getattr(hit, "accession", None) else "",
                "evalue": float(hit.evalue),
                "score": float(hit.score),
                "bias": float(hit.bias),
                "description": hit.description.decode("utf-8") if getattr(hit, "description", None) else "",
                "rank": None,
                "search_alphabet": search_alphabet,
                "source_profile": source_profile,
            }
            chain_rows.append(row)

        chain_rows.sort(key=lambda r: (r["evalue"], -r["score"], r["target_name"]))
        for rank, row in enumerate(chain_rows, start=1):
            row["rank"] = rank
        all_rows.extend(chain_rows)
        chain_summary_rows.append(
            _make_chain_summary_row(
                query_name=query_name,
                query_kind=query_kind,
                query_len=query_len,
                search_alphabet=search_alphabet,
                source_profile=source_profile,
                chain_rows=chain_rows,
            )
        )
        for row in chain_rows[: max(topk, 1)]:
            best_rows.append(dict(row))
            target_name = str(row["target_name"])
            if target_name not in selected_names and target_name in target_map:
                selected_names.add(target_name)
                selected_records.append(target_map[target_name])

    all_rows = _sort_hit_rows(all_rows)
    best_rows = _sort_hit_rows(best_rows)
    chain_summary_rows = sorted(
        chain_summary_rows,
        key=lambda r: (str(r.get("query_kind", "")), str(r.get("query_name", ""))),
    )

    print(f"# HMM search {mol}: writing outputs to {mol_dir}")
    _write_tsv(mol_dir / "all_hits.tsv", all_rows, HIT_FIELDNAMES)
    _write_tsv(mol_dir / "best_hits.tsv", best_rows, HIT_FIELDNAMES)
    _write_tsv(mol_dir / "chain_summary.tsv", chain_summary_rows, CHAIN_SUMMARY_FIELDNAMES)
    _write_fasta(mol_dir / "selected.fa", selected_records)
    return all_rows, best_rows, chain_summary_rows, selected_records


def add_args(parser):
    parser.usage = (
        "%(prog)s --input-dir INPUT_DIR --protein-fasta PROTEIN_FASTA --na-fasta NA_FASTA "
        "[--output-dir OUTPUT_DIR] [--mol protein|na|all] [--evalue E] [--cpus N]"
    )
    parser.add_argument("--input-dir", "-i", required=True, help="EM3DFold build output directory")
    parser.add_argument("--protein-fasta", "-p", help="Protein FASTA database")
    parser.add_argument("--na-fasta", "-n", help="Nucleic-acid FASTA database")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--mol", choices=("protein", "na", "all"), default="all", help="Which molecule types to search")
    parser.add_argument("--evalue", type=float, default=10.0, help="E-value cutoff")
    parser.add_argument("--F1", type=float, default=0.02)
    parser.add_argument("--F2", type=float, default=0.001)
    parser.add_argument("--F3", type=float, default=1e-5)
    parser.add_argument("--T", type=float, default=None)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--topk", type=int, default=3, help="Top-k hits per query chain")
    parser.add_argument("--min-chain-len", type=int, default=6, help="Minimum query chain length to search")
    parser.add_argument("--max-target-len", type=int, default=100000, help="Skip target sequences longer than this limit")
    parser.add_argument("--no-write-a2m", action="store_true", help="Do not write A2M files")
    parser.add_argument("--no-write-report", action="store_true", help="Do not write per-chain HMM search reports")
    return parser


def main(args):
    input_dir = Path(abspath(args.input_dir))
    output_dir = Path(abspath(args.output_dir)) if args.output_dir else input_dir / "hmm_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    begin_time = time.time()
    progress(f"EM3DFold begin at {_format_wall_time(begin_time)}")
    profile_dirs = _collect_profile_dirs(str(input_dir))
    active_stages = ["prepare"]
    if args.mol in {"protein", "all"}:
        active_stages.append("protein search")
    if args.mol in {"na", "all"}:
        active_stages.append("na search")
    active_stages.append("write outputs")

    for line in HMM_SEARCH_CONTACT_LINES:
        progress(line)
    stage_start = _announce_stage(1, len(active_stages), active_stages[0])
    progress(f"HMM search input dir: {input_dir}")
    progress(f"HMM search output dir: {output_dir}")
    progress(f"HMM search profile dirs: {', '.join(f'{k}={v}' for k, v in profile_dirs.items()) or 'none'}")
    _finish_stage(stage_start)
    if not profile_dirs:
        raise FileNotFoundError(f"No final HMM profiles found under {input_dir / 'hmm_profiles'}")

    summary: Dict[str, object] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "mol": args.mol,
        "protein_fasta": args.protein_fasta,
        "na_fasta": args.na_fasta,
        "search_params": {
            "evalue": args.evalue,
            "F1": args.F1,
            "F2": args.F2,
            "F3": args.F3,
            "T": args.T,
            "cpus": args.cpus,
            "topk": args.topk,
            "min_chain_len": args.min_chain_len,
            "max_target_len": args.max_target_len,
        },
        "profile_dirs": {key: str(value) for key, value in profile_dirs.items()},
    }

    all_rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []
    chain_summary_all: List[Dict[str, object]] = []
    selected_all: List[Tuple[str, str, str]] = []

    if args.mol in {"protein", "all"}:
        stage_idx = active_stages.index("protein search") + 1
        stage_start = _announce_stage(stage_idx, len(active_stages), active_stages[stage_idx - 1])
        if not args.protein_fasta:
            raise ValueError("--protein-fasta is required when --mol protein or --mol all")
        protein_dir = profile_dirs.get("protein") or profile_dirs.get("all")
        if protein_dir is None:
            raise FileNotFoundError("No protein HMM profiles found in the final profile directory")
        progress(f"HMM search protein: loading query HMMs from {protein_dir}")
        protein_entries = _filter_by_length(_load_hmms(protein_dir, "protein"), args.min_chain_len)
        progress(f"HMM search protein: retained {len(protein_entries)} query chains with min-chain-len={args.min_chain_len}")
        if not protein_entries:
            raise RuntimeError(f"No protein query chains passed min-chain-len={args.min_chain_len}")
        protein_rows, protein_best, protein_chain_summary, protein_selected = _search_one_mol(
            mol="protein",
            hmm_entries=protein_entries,
            fasta_path=args.protein_fasta,
            alphabet=pyhmmer.easel.Alphabet.amino(),
            output_dir=output_dir,
            evalue=args.evalue,
            F1=args.F1,
            F2=args.F2,
            F3=args.F3,
            T=args.T,
            cpus=args.cpus,
            topk=args.topk,
            write_a2m=(not args.no_write_a2m),
            write_report=(not args.no_write_report),
            max_target_len=args.max_target_len,
        )
        all_rows.extend(protein_rows)
        best_rows.extend(protein_best)
        chain_summary_all.extend(protein_chain_summary)
        selected_all.extend(protein_selected)
        summary["protein_query_count"] = len(protein_entries)
        summary["protein_query_with_hits_count"] = sum(int(row.get("num_hits", 0) > 0) for row in protein_chain_summary)
        summary["protein_hit_count"] = len(protein_rows)
        summary["protein_selected_count"] = len(protein_selected)
        progress(
            f"HMM search protein: queries={len(protein_entries)} "
            f"queries_with_hits={summary['protein_query_with_hits_count']} hits={len(protein_rows)} selected={len(protein_selected)}"
        )
        _finish_stage(stage_start)

    if args.mol in {"na", "all"}:
        stage_idx = active_stages.index("na search") + 1
        stage_start = _announce_stage(stage_idx, len(active_stages), active_stages[stage_idx - 1])
        if not args.na_fasta:
            raise ValueError("--na-fasta is required when --mol na or --mol all")
        na_dir = profile_dirs.get("na") or profile_dirs.get("all")
        if na_dir is None:
            raise FileNotFoundError("No nucleic-acid HMM profiles found in the final profile directory")
        progress(f"HMM search na: loading query HMMs from {na_dir}")
        na_entries = _filter_by_length(_load_hmms(na_dir, "na"), args.min_chain_len)
        progress(f"HMM search na: retained {len(na_entries)} query chains with min-chain-len={args.min_chain_len}")
        if not na_entries:
            raise RuntimeError(f"No na query chains passed min-chain-len={args.min_chain_len}")
        na_rows, na_best, na_chain_summary, na_selected = _search_one_mol(
            mol="na",
            hmm_entries=na_entries,
            fasta_path=args.na_fasta,
            alphabet=pyhmmer.easel.Alphabet.rna(),
            output_dir=output_dir,
            evalue=args.evalue,
            F1=args.F1,
            F2=args.F2,
            F3=args.F3,
            T=args.T,
            cpus=args.cpus,
            topk=args.topk,
            write_a2m=(not args.no_write_a2m),
            write_report=(not args.no_write_report),
            max_target_len=args.max_target_len,
        )
        all_rows.extend(na_rows)
        best_rows.extend(na_best)
        chain_summary_all.extend(na_chain_summary)
        selected_all.extend(na_selected)
        summary["na_query_count"] = len(na_entries)
        summary["na_query_with_hits_count"] = sum(int(row.get("num_hits", 0) > 0) for row in na_chain_summary)
        summary["na_hit_count"] = len(na_rows)
        summary["na_selected_count"] = len(na_selected)
        progress(
            f"HMM search na: queries={len(na_entries)} "
            f"queries_with_hits={summary['na_query_with_hits_count']} hits={len(na_rows)} selected={len(na_selected)}"
        )
        _finish_stage(stage_start)

    dedup_selected_all: List[Tuple[str, str, str]] = []
    seen_selected_names = set()
    for record in selected_all:
        if record[0] in seen_selected_names:
            continue
        seen_selected_names.add(record[0])
        dedup_selected_all.append(record)

    all_rows = _sort_hit_rows(all_rows)
    best_rows = _sort_hit_rows(best_rows)
    chain_summary_all = sorted(
        chain_summary_all,
        key=lambda r: (str(r.get("query_kind", "")), str(r.get("query_name", ""))),
    )
    summary["total_hit_count"] = len(all_rows)
    summary["total_best_hit_count"] = len(best_rows)
    summary["selected_all_count"] = len(dedup_selected_all)

    stage_idx = len(active_stages)
    stage_start = _announce_stage(stage_idx, len(active_stages), active_stages[-1])
    progress(f"HMM search: writing merged outputs to {output_dir}")
    _write_tsv(output_dir / "all_hits.tsv", all_rows, HIT_FIELDNAMES)
    _write_tsv(output_dir / "best_hits.tsv", best_rows, HIT_FIELDNAMES)
    _write_tsv(output_dir / "chain_summary.tsv", chain_summary_all, CHAIN_SUMMARY_FIELDNAMES)
    _write_fasta(output_dir / "selected_all.fa", dedup_selected_all)
    with open(output_dir / "search_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    progress(
        f"HMM search summary: total_hits={len(all_rows)} "
        f"total_best_hits={len(best_rows)} selected_all={len(dedup_selected_all)}"
    )
    _finish_stage(stage_start)
    progress(f"HMM search complete: {output_dir}")
    progress(f"EM3DFold end at {_format_wall_time()}")
    return 0
