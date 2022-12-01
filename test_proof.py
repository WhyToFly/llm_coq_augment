import asyncio
import os
import pathlib

from typing import Iterable

import pycoq.opam
import pycoq.common
import pycoq.agent

import shutil

import json

from concurrent.futures import ProcessPoolExecutor

def create_conf():
    # create default coq context for evaluation of a theorem
    coq_ctxt = pycoq.common.CoqContext(pwd=os.getcwd(),
                                       executable='',
                                       target='serapi_shell')
    cfg = pycoq.opam.opam_serapi_cfg(coq_ctxt)
    return cfg

# I do not like asyncio because I don't understand it
def test_coq_aio(params):
    cfg, theorem, proof, debug = params

    try:
        error = asyncio.run(asyncio.wait_for(test_coq(cfg, theorem, proof, debug), timeout=15))
    except Exception as e:
        error = str(e)

    return error

# function to test a lemma with or without proof
async def test_coq(cfg, theorem, proof, debug=False):
    prop = theorem["env"] + "\n" + theorem["lemma"]
    if (proof is not None) and (len(proof) > 0):
        prop += "\n" + proof

    if debug:
        print("\n\n Testing:\n" + prop)

    async with pycoq.serapi.CoqSerapi(cfg) as coq:
        # execute proposition of the theorem
        try:
            _, _, coq_exc, _ = await coq.execute(prop)
        except Exception as e:
            coq_exc = str(e)
        if coq_exc:
            if debug:
                print("ERROR IN " + theorem["filepath"])
                #print()
                #print(prop)
                #print()
                print(coq_exc)
                print("-----------------------------------")
            return coq_exc
        else:
            return None

# test if there is an exception running the env and lemma 
# (in the list of dictionaries) or if they are valid
def test_lemmas(theorems: Iterable, debug=False):
    cfg = create_conf()

    # execute checking of lemmas on multiple CPU cores
    params = [(cfg, theorem, None, debug) for theorem in theorems]
    with ProcessPoolExecutor() as executor:
        errors = executor.map(test_coq_aio, params)

    return errors

# test if there is an exception running the env, lemma and proof
# (in the list of dictionaries) or if they are valid
def test_proofs(theorems: Iterable, debug=False):
    cfg = create_conf()

    # execute checking of lemmas on multiple CPU cores
    params = [(cfg, theorem, theorem["proof"], debug) for theorem in theorems]
    with ProcessPoolExecutor() as executor:
        errors = executor.map(test_coq_aio, params)

    return errors

# remove all files that contain envs/lemmas that are not valid (as verified by coq)
def filter_theorems(theorems: Iterable):
    errors = test_lemmas(theorems)

    count = 0
    for i, error in enumerate(errors):
        if error is not None:
            print("removing " + str(theorems[i]["filepath"]))
            # there is an error with this theorem; we should remove it
            os.remove(theorems[i]["filepath"])
            count += 1

    print(str(count) + " out of " + str(len(theorems)) + " theorems removed.")

# remove all files that contain envs/lemmas/proofs that are not valid (as verified by coq)
def filter_proofs(theorems: Iterable):
    errors = test_proofs(theorems)

    count = 0
    for i, error in enumerate(errors):
        if error is not None:
            print("removing " + str(theorems[i]["filepath"]))
            # there is an error with this theorem; we should remove it
            os.remove(theorems[i]["filepath"])
            count += 1

    print(str(count) + " out of " + str(len(theorems)) + " theorems removed.")

def main():
    # filter out all theorems that our coq endpoint cannot run for one reason or another
    proofs = []

    for json_file in pathlib.Path("./coq_PL/").rglob('*.json'):
        with open(json_file, "r") as f:
            theorem = json.load(f)
            theorem["filepath"] = str(json_file)
            proofs.append(theorem)
    
    for json_file in pathlib.Path("./coq_theorems/").rglob('*.json'):
        with open(json_file, "r") as f:
            theorem = json.load(f)
            theorem["filepath"] = str(json_file)
            proofs.append(theorem)

    filter_theorems(proofs)

    # filter out all theorems that contain proofs that are not seen as valid by our coq endpoint - used for LLMs afterwards
    proofs = []

    shutil.rmtree("./proven_theorems/")
    shutil.copytree("./coq_theorems/", "./proven_theorems/")

    for json_file in pathlib.Path("./proven_theorems/").rglob('*.json'):
        with open(json_file, "r") as f:
            theorem = json.load(f)
            theorem["filepath"] = str(json_file)
            proofs.append(theorem)

    filter_proofs(proofs)

if __name__ == "__main__":
    main()