# What is it?

The project is to adopt famous llama.cpp to fmsh backend, which has an ARM processor and a NPU. 
Currently llama.cpp could run on it at pure-cpu mode. The next step is to provide NPU accel to llama.

# Tools to build

We are developing on a amd64 platform, so you could compile and test it on amd64 for errors, but do expect different behavior. 
To cross-compile the project to this target and run on it, use `scripts/build_llama_arm.sh` with args, which will run compiled llama-cli on remote devices with given args, e.g. `scripts/build_llama_arm.sh --list-devices`.

## Test with real payload

To load and test a LLM, use

```bash
scripts/build_llama_arm.sh -m ../Qwen3.5-0.8B-Q4_K_M.gguf -p "Intro urself in one sentence" -c 1024 --reasoning-budget 0
```

This will take a long time!
# Docs

The nessary docs about fmsh backend are shown in ./docs folder.

# Reference

A fully-functional yolov5 adoption to fmsh backend is available at ./ref_proj.