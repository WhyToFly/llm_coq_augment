import sys
import os
import time
import random
import pathlib
import json
import test_proof
import coq_parser
from os.path import join
import shutil

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "EleutherAI/gpt-neo-125M"
"""
model_path = "gpt-neo-125M-coq/ckpt-28800"
model = AutoModelForCausalLM.from_pretrained(model_path, from_flax=True).to(device)
"""

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token


def get_prompt_part(rand_theorem, with_proof):
    if with_proof:
        return rand_theorem["env"] + "\n" + rand_theorem["lemma"] + "\n" \
                    + rand_theorem["proof"] + "\n\n" + "----------------------" + "\n\n"
    else:
        return rand_theorem["env"] + "\n" + rand_theorem["lemma"] \
                    + "\n\n" + "----------------------" + "\n\n"

# have GPT generate more theorems
def gen_theorems(proven_theorems, n, out_dir, num_iterations):
    for it in range(num_iterations):
        # generate theorems: chain together other random theorems to show model what we expect
        prompt = ""
        # make sure we don't make this too long; should be shorter than context window of the model
        # and leave enough room for prediction

        # max length for generated text
        max_theorem_len = 512 

        rand_theorem = random.choice(proven_theorems)
        # according to OpenAI, a token is about 4 chars; let's assume it's 3 to be safe
        while (len(prompt) + len(get_prompt_part(rand_theorem, True)))/3 - max_theorem_len < 1500:
            prompt = get_prompt_part(rand_theorem, True) + prompt

            rand_theorem = random.choice(proven_theorems)

        success = False

        # try to free up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # potential issue with GPT-Neo: max len is a bunch smaller than Codex (2048 vs. 8000)
        # It is actually too small to fit some envs
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500-max_theorem_len).input_ids.to(device)
        start = len(input_ids[0])

        try:
            res = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
                temperature=0.9,
                num_beams=1,
                no_repeat_ngram_size=None,
                repetition_penalty=None,
                num_return_sequences=n,
            )

            success = True
        except Exception as e:
            print(e)
        # create proof dicts for all responses

        if success:
            i = 0
            for text in res:
                # format so our parser returns a result
                text = tokenizer.decode(text[start:])

                # check if proof ended
                if not "Qed." in text:
                    text += "\nQed."
                else:
                    text = text.split("Qed.")[0] + "Qed."

                # try to extract a theorem
                lemma_name, theorem = next(coq_parser.extract_theorems(text), (None, None))

                if theorem is not None:
                    os.makedirs(out_dir, exist_ok=True)
                    with open(join(out_dir, lemma_name + time.strftime("%Y%m%d-%H%M%S") + str(i) + ".json"), "w") as f:
                        json.dump(theorem, f)
                    i += 1

# have gpt generate a proof for the theorem
# prompt: randomly chosen solutions to theorems
def gen_proof(proven_theorems, theorem, n):
    # generate prompt: chain together other random proofs to show model what we expect
    prompt = theorem["env"] + "\n" + theorem["lemma"] + "\n" + "Proof." + "\n"
    # make sure we don't make this too long; should be shorter than context window of the model
    # and leave enough room for prediction

    # max length for generated text
    max_proof_len = 256 

    rand_theorem = random.choice(proven_theorems)
    # according to OpenAI, a token is about 4 chars
    while (len(prompt) + len(get_prompt_part(rand_theorem, True)))/4 - max_proof_len < 1500:
        prompt = get_prompt_part(rand_theorem, True) + prompt
        rand_theorem = random.choice(proven_theorems)

    success = False


    # try to free up CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # potential issue with GPT-Neo: max len is a bunch smaller than Codex (2048 vs. 8000)
    # It is actually too small to fit some envs
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500-max_proof_len).input_ids.to(device)
    start = len(input_ids[0])

    try:
        res = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            temperature=0.8,
            num_beams=1,
            no_repeat_ngram_size=None,
            repetition_penalty=None,
            num_return_sequences=n,
        )

        success = True
    except Exception as e:
        print(e)

    proofs = []

    # create proof dicts for all responses

    if success:
        for text in res:
            text = "\n" + "Proof." + "\n" + tokenizer.decode(text[start:], skip_special_tokens=True)

            # check if proof ended
            if not "Qed." in text:
                text += "\nQed."
            else:
                text = text.split("Qed.")[0] + "\nQed."

            proof = theorem.copy()

            proof["proof"] = text
            proofs.append(proof)

    return proofs

# get n proofs for the given theorem from GPT-Neo; if one of them is correct, save it to output_path
def gpt_prove_and_test(cfg, prompt_proofs, theorem, output_path, n):
    proofs = gen_proof(prompt_proofs, theorem, n)

    # test generated proofs for correctness
    errors = test_proof.test_proofs(proofs, debug=False)

    # find shortest correct proof (original paper tries to find more but let's leave it at 1 for now)
    solved = False
    shortest = 100000000000
    shortest_ind = 0
    for i, error in enumerate(errors):
        # if the proof is correct: save it
        if error is None:
            if len(proofs[i]["proof"]) < shortest:
                solved = True
                shortest = len(proofs[i]["proof"])
                shortest_ind = i

    if not solved:
        print("GPT-Neo didn't solve " + theorem["filepath"])
        if len(proofs) > 0:
            os.makedirs("/".join(join(join(output_path, "unsolved"), proofs[shortest_ind]["filepath"]).split("/")[:-1]), exist_ok=True)
            with open(join(join(output_path, "unsolved"), proofs[shortest_ind]["filepath"]), "w") as f:
                json.dump(proofs[shortest_ind], f)
    else:
        print("GPT-Neo solved " + proofs[shortest_ind]["filepath"])
        os.makedirs("/".join(join(join(output_path, "solved"), proofs[shortest_ind]["filepath"]).split("/")[:-1]), exist_ok=True)
        with open(join(join(output_path, "solved"), proofs[shortest_ind]["filepath"]), "w") as f:
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
    
    success_count = 0

    for theorem in theorems:
        # generate a proof, test it, save it if it proves the theorem and increase the success counter
        if gpt_prove_and_test(cfg, prompt_proofs, theorem, output_path, n):
            success_count += 1

    print("GPT-Neo proved " + str(success_count) + " out of " + str(len(theorems)) + " theorems.")

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
        print("Generating " + str(num_iterations) + " * " + str(n) + " theorems.")

        out_path = join(output_path, time.strftime("%Y%m%d-%H%M%S"))

        try:
            # generate theorems
            gen_theorems(prompt_proofs, n, out_path, num_iterations)

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

# how many proofs should GPT generate per theorem?
NUM_GENERATIONS = 24

def main():
    test_path = "coq_PL/"
    prompt_proofs_path = "proven_theorems/"
    if sys.argv[1] == "eval":
        output_path = "GPT_proofs/coq_PL/" + model_path.split("/")[-1]
        test_model(test_path, prompt_proofs_path, output_path, NUM_GENERATIONS)
    elif sys.argv[1] == "generate":
        output_path = "GPT_synthetic_proofs/"
        gen_synthetic_theorems(prompt_proofs_path, output_path, NUM_GENERATIONS, 5)
    else:
        print("Please specify either argument 'eval' (to evaluate the model on the PL class theorems) or 'generate' (to let the model generate synthetic training data)")

if __name__ == "__main__":
    main()
