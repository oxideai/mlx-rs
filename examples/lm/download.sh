#!/bin/bash

model_id="mlx-community/Qwen3-4B-bf16"

huggingface-cli download $model_id --local-dir ./cache/Qwen3-4B-bf16