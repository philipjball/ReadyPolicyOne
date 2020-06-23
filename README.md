# Ready Policy One (RP1)

Code to complement ["Ready Policy One: World Building through Active Learning"](https://arxiv.org/abs/2002.02693).

## General instructions

All the experiments run in this paper exist in the `args_yml` directory. On the machines we trained on, we could run 5 seeds concurrently, hence the macro-level script `run_experiments.py` launches 5 at once, with a binary flag to toggle seeds 0-4 or 5-9.

To run the `HalfCheetah` Ready Policy One experiments for seeds 5-9, type the following:

`python run_experiments.py --yaml ./args_yml/main_exp/halfcheetah-rp1.yml --seeds5to9`

## FAQs

### Why is model free running so slowly?

Two reasons: 1) It is non-parallelised; 2) This code tries to find GPUs where possible, try forcing it to run on CPU

## Acknowledgements

The authors acknowledge Nikhil Barhate for his [PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch) repo. The `ppo.py` file here is a heavily modified version of this code.
