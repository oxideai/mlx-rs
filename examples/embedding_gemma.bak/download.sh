#!/bin/bash

model_id="mlx-community/embeddinggemma-300m-bf16"

huggingface-cli download $model_id --local-dir ./cache/embeddinggemma-300m-bf16