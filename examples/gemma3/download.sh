#!/bin/bash

model_id="mlx-community/gemma-3-270m-bf16"

huggingface-cli download $model_id --local-dir ./cache/gemma-3-270m-bf16