import sys
import os
import openai
import time
import random
import pathlib
import json
import test_proof
import coq_parser
from os.path import join
import shutil
import traceback

from concurrent.futures import ProcessPoolExecutor

key_index = 0
keys = [YOUR OPENAI API KEY(S)]

def get_prompt_part(rand_theorem, with_proof):
    if with_proof:
        return rand_theorem["env"] + "\n" + rand_theorem["lemma"] + "\n" \
                    + rand_theorem["proof"] + "\n\n" + "----------------------" + "\n\n"
    else:
        return rand_theorem["env"] + "\n" + rand_theorem["lemma"] \
                    + "\n\n" + "----------------------" + "\n\n"

# generate theorems on multiple CPUs
def codex_gen_theorems_helper(params):
    proven_theorems, n, api_key, out_dir, num_iterations = params
    gen_theorems(proven_theorems, n, api_key, out_dir, num_iterations)

# have codex generate more theorems
def gen_theorems(proven_theorems, n, api_key, out_dir, num_iterations):
    for it in range(num_iterations):
        # generate theorems: chain together other random theorems to show model what we expect
        prompt = ""
        # make sure we don't make this too long; should be shorter than context window of the model
        # and leave enough room for prediction

        # max length for generated text
        max_theorem_len = 512 

        rand_theorem = random.choice(proven_theorems)
        # according to OpenAI, a token is about 4 chars; let's assume it's 3 to be safe
        while (len(prompt) + len(get_prompt_part(rand_theorem, True)))/3 - max_theorem_len < 4000:
            prompt = get_prompt_part(rand_theorem, True) + prompt

            rand_theorem = random.choice(proven_theorems)

        success = False
        tries = 0

        while (not success) and (tries < 2):
            try:
                openai.api_key = api_key
                tries += 1
                res = openai.Completion.create(
                    model="code-davinci-002",
                    prompt=prompt,
                    max_tokens=max_theorem_len,
                    temperature=0.9,
                    stop="Qed.",
                    n=n
                    )
                success = True
            except Exception as e:
                print(e)
                time.sleep(60)

        # create theorem dicts for all responses
        i = 0
        if success:
            for text in res["choices"]:
                # format so our parser returns a result
                text = text["text"] + "Qed."

                # try to extract a theorem
                lemma_name, theorem = next(coq_parser.extract_theorems(text), (None, None))

                if theorem is not None:
                    os.makedirs(out_dir, exist_ok=True)
                    with open(join(out_dir, lemma_name + time.strftime("%Y%m%d-%H%M%S") + str(i) + ".json"), "w") as f:
                        json.dump(theorem, f)
                    i += 1

# have codex generate a proof for the theorem
# prompt: randomly chosen solutions to theorems
def gen_proof(proven_theorems, theorem, n, api_key):
    # generate prompt: chain together other random proofs to show model what we expect
    prompt = theorem["env"] + "\n" + theorem["lemma"] + "\n" + "Proof." + "\n"
    # make sure we don't make this too long; should be shorter than context window of the model
    # and leave enough room for prediction

    # max length for generated text
    max_proof_len = 256 

    rand_theorem = random.choice(proven_theorems)

    # according to OpenAI, a token is about 4 chars; let's assume it's 3 to be safe
    while (len(prompt) + len(get_prompt_part(rand_theorem, True)))/3 - max_proof_len < 6000:
        prompt = get_prompt_part(rand_theorem, True) + prompt
        rand_theorem = random.choice(proven_theorems)

    success = False
    tries = 0

    while (not success) and (tries < 2):
        try:
            openai.api_key = api_key
            tries += 1
            res = openai.Completion.create(
                model="code-davinci-002",
                prompt=prompt,
                max_tokens=max_proof_len,
                temperature=0.8,
                stop="Qed.",
                n=n
                )
            success = True
        except Exception as e:
            print(e)
            time.sleep(30)

    proofs = []

    # create proof dicts for all responses
    if success:
        for text in res["choices"]:
            text = "\n" + "Proof." + "\n" + text["text"] + "\nQed."
            proof = theorem.copy()
            proof["proof"] = text
            proofs.append(proof)

    return proofs

# helper function to pass all parameters in one tuple (for multi CPU)
def codex_prove_and_test_helper(params):
    return codex_prove_and_test(*params)

# get n proofs for the given theorem from codex; if one of them is correct, save it to output_path
def codex_prove_and_test(cfg, prompt_proofs, theorem, output_path, n, api_key):
    proofs = gen_proof(prompt_proofs, theorem, n, api_key)

    # test generated proofs for correctness
    errors = test_proof.test_proofs(proofs, debug=False)

    # find shortest correct proof (original paper tries to find more but let's leave it at 1 for now)
    solved = False
    shortest = 100000000000
    shortest_ind = -1
    for i, error in enumerate(errors):
        # if the proof is correct: save it
        if error is None:
            if len(proofs[i]["proof"]) < shortest:
                solved = True
                shortest = len(proofs[i]["proof"])
                shortest_ind = i

    if not solved:
        print("Codex didn't solve " + theorem["filepath"])
    else:
        print("Codex solved " + proofs[shortest_ind]["filepath"])
        os.makedirs("/".join(join(output_path, proofs[shortest_ind]["filepath"]).split("/")[:-1]), exist_ok=True)
        with open(join(output_path, proofs[shortest_ind]["filepath"]), "w") as f:
            json.dump(proofs[shortest_ind], f)

    return solved

# test model on inclass/assignment files
def test_model(test_path, prompt_proofs_path, output_path, n):
    cfg = test_proof.create_conf()

    prompt_proofs = []
    theorems = []

    for json_file in pathlib.Path(prompt_proofs_path).rglob('*.json'):
        with open(json_file, "r") as f:
            theorem = json.load(f)
            theorem["filepath"] = str(json_file)[len(prompt_proofs_path):]
            prompt_proofs.append(theorem)

    for json_file in pathlib.Path(test_path).rglob('*.json'):
        with open(json_file, "r") as f:
            theorem = json.load(f)
            theorem["filepath"] = str(json_file)[len(test_path):]
            theorems.append(theorem)
    
    # distribute proof generation/checking over all CPU cores
    params = [(cfg, prompt_proofs, theorems[i], output_path, n, keys[i % len(keys)]) for i in range(len(theorems))]

    with ProcessPoolExecutor(min(os.cpu_count(), len(keys))) as executor:
        successes = executor.map(codex_prove_and_test_helper, params)

    success_count = 0
    for succ in successes:
        if succ:
            success_count += 1

    print("Codex proved " + str(success_count) + " out of " + str(len(theorems)) + " theorems.")

# generate more theorems based on the existing ones
def gen_synthetic_theorems(prompt_proofs_path, output_path, n, num_iterations):
    i = 1
    cfg = test_proof.create_conf()
    prompt_proofs = []

    for json_file in pathlib.Path(prompt_proofs_path).rglob('*.json'):
        with open(json_file, "r") as f:
            theorem = json.load(f)
            theorem["filepath"] = str(json_file)[len(prompt_proofs_path):]
            prompt_proofs.append(theorem)
    while True:
        print("Iteration " + str(i) + ".")
        print("Generating " + str(num_iterations) + " * " + str(n) + " theorems for each key.")

        out_path = join(output_path, time.strftime("%Y%m%d-%H%M%S"))

        try:
            # distribute theorem generation over all CPU cores
            params = [(prompt_proofs, n, key, out_path, num_iterations) for key in keys]

            with ProcessPoolExecutor(min(os.cpu_count(), len(keys))) as executor:
                executor.map(codex_gen_theorems_helper, params)

            generated = []

            for json_file in pathlib.Path(out_path).rglob('*.json'):
                with open(json_file, "r") as f:
                    theorem = json.load(f)
                    theorem["filepath"] = str(json_file)
                    generated.append(theorem)

            print("Fitering out bad theorems.")
            # remove proofs that don't work
            test_proof.filter_proofs(generated)
            # remove dir if no proofs remain in it
            if len(os.listdir(out_path)) == 0:
                shutil.rmtree(out_path, ignore_errors=False)

            i += 1
        except KeyboardInterrupt as e:
            print("Removing last generations folder (not verified yet).")
            shutil.rmtree(out_path)
            raise e

# how many proofs should Codex generate per theorem?
NUM_GENERATIONS = 24

def main():
    test_path = "coq_PL/"
    prompt_proofs_path = "proven_theorems/"
    if sys.argv[1] == "eval":
        output_path = "codex_proofs/coq_theorems/"
        test_model(test_path, prompt_proofs_path, output_path, NUM_GENERATIONS)
    elif sys.argv[1] == "generate":
        output_path = "codex_synthetic_proofs/"
        gen_synthetic_theorems(prompt_proofs_path, output_path, 4, 4)
    else:
        print("Please specify either argument 'eval' (to evaluate the model on the PL class theorems) or 'generate' (to let the model generate synthetic training data)")

if __name__ == "__main__":
    main()
