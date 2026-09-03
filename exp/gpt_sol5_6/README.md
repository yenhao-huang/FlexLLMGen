# GPT-5.6 Sol placement optimizer

This experiment makes one bounded Responses API call to `gpt-5.6-sol` and asks
for up to five placement candidates. The response uses Structured Outputs and
is validated locally against the supplied GPU/CPU limits before it is saved.
It does not execute a proposed configuration automatically.

Install the optional SDK and provide an API key:

```bash
pip install -e '.[sol]'
export OPENAI_API_KEY=...
```

After the deterministic baseline/search has produced results, run:

```bash
python exp/gpt_sol5_6/optimize_placement.py \
  --gpu-free-gib 22 22 --cpu-available-gib 52
```

The default call uses high reasoning effort, stores no API-side response state,
and caps output at 2,000 tokens. Review `proposal.json`, benchmark its candidates
with `python -m flexllmgen.hf_opt`, and compare measured throughput; a proposal
is a hypothesis, not a result.
