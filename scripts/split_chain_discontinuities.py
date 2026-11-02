#!/usr/bin/env python3
"""Split mmCIF chains at impossible peptide-backbone gaps."""

import argparse
import string

import numpy as np
from Bio.PDB import Chain, MMCIFIO, MMCIFParser, Model, Structure


def _chain_ids():
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    for chain_id in alphabet:
        yield chain_id
    index = 1
    while True:
        yield f"X{index}"
        index += 1


def _needs_split(previous, current, max_cn_distance, max_ca_distance):
    if "C" not in previous or "N" not in current:
        return True, float("nan"), float("nan")
    cn_distance = float(np.linalg.norm(previous["C"].coord - current["N"].coord))
    ca_distance = float("nan")
    if "CA" in previous and "CA" in current:
        ca_distance = float(np.linalg.norm(previous["CA"].coord - current["CA"].coord))
    split = cn_distance > max_cn_distance
    if np.isfinite(ca_distance):
        split = split or ca_distance > max_ca_distance
    return split, cn_distance, ca_distance


def split_structure(input_path, output_path, max_cn_distance, max_ca_distance):
    source = MMCIFParser(QUIET=True).get_structure("input", input_path)
    source_model = next(source.get_models())
    output = Structure.Structure("output")
    output_model = Model.Model(0)
    output.add(output_model)
    chain_id_iter = _chain_ids()
    num_input_chains = 0
    num_output_chains = 0
    num_residues = 0
    num_splits = 0

    for source_chain in source_model:
        residues = [residue for residue in source_chain if residue.id[0] == " "]
        if not residues:
            continue
        num_input_chains += 1
        segments = []
        segment = [residues[0]]
        for previous, current in zip(residues[:-1], residues[1:]):
            should_split, _, _ = _needs_split(
                previous,
                current,
                max_cn_distance,
                max_ca_distance,
            )
            if should_split:
                segments.append(segment)
                segment = []
                num_splits += 1
            segment.append(current)
        segments.append(segment)

        for segment in segments:
            output_chain = Chain.Chain(next(chain_id_iter))
            output_model.add(output_chain)
            for residue_index, residue in enumerate(segment):
                copied = residue.copy()
                copied.id = (" ", residue_index, " ")
                output_chain.add(copied)
                num_residues += 1
            num_output_chains += 1

    writer = MMCIFIO()
    writer.set_structure(output)
    writer.save(output_path)
    return {
        "input_chains": num_input_chains,
        "output_chains": num_output_chains,
        "residues": num_residues,
        "splits": num_splits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cn-distance", type=float, default=2.0)
    parser.add_argument(
        "--max-ca-distance",
        type=float,
        default=float("inf"),
        help="Optional CA safeguard; disabled by default because tracing is defined by C-N distance",
    )
    args = parser.parse_args()
    summary = split_structure(
        args.input,
        args.output,
        args.max_cn_distance,
        args.max_ca_distance,
    )
    print(summary)


if __name__ == "__main__":
    main()
