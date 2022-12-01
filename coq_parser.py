"""
Very simple (and naive) parser to extract training data from coq files
Assumes everything is nicely divided in lines (which is hopefully true for standard coq library)
"""

import os
from os.path import join
import json
import pathlib
import re

def extract_theorems_from_file(source, module, filename, out_dir):
    # create output dir for this file
    out_path = join(join(out_dir, module), filename[:-2])

    # load file
    filepath = join(join(source, module), filename)
    file_contents = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            file_contents = f.read()
    except Exception as e:
        print(e)
        with open(filepath, "r", encoding="latin") as f:
            file_contents = f.read()

    for lemma_name, theorem in extract_theorems(file_contents, module, filename):
        # write env, lemma and proof to file
        os.makedirs(out_path, exist_ok=True)
        outfile = join(out_path, lemma_name + ".json")
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(theorem, f)

def extract_theorems(theorem_text, module="", filename=""):
    file_contents = theorem_text.split("\n")

    # find out if there is a starting comment block (copyright, license etc.)
    if file_contents[0].strip().startswith("(*****"):
        while file_contents[0].strip().startswith("(*"):
            file_contents.pop(0)

    # parse file and structure into sections; 
    # sections can be nested and we only want to remember 
    # things that are in the sections we are currently in

    # list that keeps starts of sections; any time we read a "Section" command, 
    # we add the current number of lines here (without Lemmas/Theorems).
    # when we read an "End", we remove the last start and all lines that come after it.
    section_starts = []

    env_lines = []
    # start by importing the file we are currently working with - that way, 
    # coq can (hopefully?) use previous lemmas from that file 
    if len(module) > 0:
        env_lines.append("Require Import " + filename[:-2].split("/")[-1] + ".")

    in_ltac = False
    in_lemma = False
    in_proof = False
    in_comment = False
    in_hint = False

    lemma_name = ""
    lemma_lines = []
    proof_lines = []

    for line in file_contents:
        stripped = line.strip()

        # find comments
        if stripped.startswith("(*"):
            in_comment = True
        if stripped.endswith("*)"):
            in_comment = False

        # don't do any detections/skipping in comments (or after just closing one)
        elif not in_comment:
            # filter out Hints
            if stripped.startswith("Hint "):
                in_hint = True
            # find end of Hint
            if in_hint:
                if stripped.endswith("."):
                    in_hint = False
                continue

            # filter out Ltacs
            if stripped.startswith("Ltac "):
                in_ltac = True
            # find end of Ltac
            if in_ltac:
                if stripped.endswith("."):
                    in_ltac = False
                continue

            # find start of a section
            if stripped.startswith("Section "):
                section_starts.append((stripped.split(" ")[-1], len(env_lines)))
                continue

            # find end of a section; adjust local context
            if stripped.startswith("End "):
                # figure out if end of section or module
                if (len(section_starts) > 0) and (stripped.split(" ")[-1] == section_starts[-1][0]):
                    env_lines = env_lines[:section_starts.pop(-1)[1]]
                    continue

            # find Theorems/Lemmas
            if stripped.startswith("Lemma ") or stripped.startswith("Theorem ") \
                or stripped.startswith("Fact ") or stripped.startswith("Remark ") \
                or stripped.startswith("Corollary ") or stripped.startswith("Proposition ") \
                or stripped.startswith("Property ") or stripped.startswith("Example "):
                in_lemma = True
                lemma_name = stripped.split(" ")[1]
                lemma_lines = []

            # add to lemma and find end of theorem/lemma
            if in_lemma:
                if stripped.startswith("Proof"):
                    in_lemma = False
                    in_proof = True

                    proof_lines = []
                else:
                    lemma_lines.append(line)
                    continue
                
            # find end of proof (could either be in the format "Proof lemma." or "Proof. \n...\n Qed.")
            if in_proof:
                
                proof_lines.append(line)
                if ((stripped.startswith("Proof ") and (not stripped.startswith("Proof with"))) \
                    or stripped.endswith("Qed.") or stripped.endswith("Admitted.") \
                    or stripped.endswith("Abort.")):
                    in_proof = False

                    # create theorem dict
                    if lemma_name.endswith(":"):
                        lemma_name = lemma_name[:-1]

                    # clean up data to take up less space - GPT-Neo has a small window and 
                    # we don't want to fill it with blank space and comments 
                    # (comments might be very helpful as they offer natural language explanations; however they're also very long)

                    env = "\n".join(env_lines)
                    lemma = "\n".join(lemma_lines)
                    proof = "\n".join(proof_lines)


                    # remove comments (regex adapted from https://stackoverflow.com/questions/14596884/remove-text-between-and)
                    env = re.sub("\\(\\*.*?\\*\\)", "", env, flags=re.S)
                    lemma = re.sub("\\(\\*.*?\\*\\)", "", lemma, flags=re.S)
                    proof = re.sub("\\(\\*.*?\\*\\)", "", proof, flags=re.S)

                    # remove multiple newlines
                    env = re.sub(r'\n+', '\n', env)
                    lemma = re.sub(r'\n+', '\n', lemma)
                    proof = re.sub(r'\n+', '\n', proof)

                    yield lemma_name, {"env":env, "lemma":lemma, "proof":proof}
                    continue

        # if we didn't find and special section: just append to env
        if not (in_lemma or in_ltac or in_proof):
            env_lines.append(line)


COQ_THEORIES = "$HOME/.opam/default/lib/coq/theories/"
COQ_PLUGINS = "$HOME/.opam/default/lib/coq/plugins/"
PL_ASSIGNMENTS = "$HOME/PL_proj/PL_lectures/"
PL_LECTURES = "$HOME/PL_proj/PL_assignments/"

def main():
    # iterate over all modules, .v files in theories and plugins 
    # (make everything below into function that is called with all paths)
    # that way, we can treat test files the same way
    for source in [PL_ASSIGNMENTS, PL_LECTURES]:
        for module in [""]:#pathlib.Path(source).glob('*'):
            #module = str(module).split("/")[-1]
            for filename in pathlib.Path(join(source, module)).rglob('*.v'):
                filename = str(filename)[len(join(source, module)):]

                extract_theorems_from_file(source, module, filename, "./coq_PL/")

    for source in [COQ_THEORIES, COQ_PLUGINS]:
        for module in pathlib.Path(source).glob('*'):
            module = str(module).split("/")[-1]
            for filename in pathlib.Path(join(source, module)).rglob('*.v'):
                filename = str(filename)[len(join(source, module))+1:]

                extract_theorems_from_file(source, module, filename, "./coq_theorems/")


if __name__ == "__main__":
    main()