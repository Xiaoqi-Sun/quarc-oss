# QUARC (QUAtitative Recommendations of reaction Conditions)

Training, benchmarking and serving modules for reaction condition recommendation with QUARC, a data-driven model for predicting agents, temperature, and equivalence ratios for organic synthesis.

Unless otherwise specified, models are released under the same license as the source code (MIT license).

> [!IMPORTANT]
> This open-source version of QUARC differs from the version described in the [paper](https://chemrxiv.org/engage/chemrxiv/article-details/686809c0c1cb1ecda020efc1). The paper version requires both a reaction SMILES and a [NameRxn](https://www.nextmovesoftware.com/namerxn.html) reaction class label as input, whereas the open-source version does not rely on any proprietary reaction classification.
>
> To access the full version of QUARC used in the paper, please refer to the [quarc repo](https://github.com/coleygroup/quarc), which assumes access to NameRxn's reaction types.

## Serving

### Step 1/4: Environment Setup

First set up the url to the remote registry

```
export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core
```

Then follow the instructions below to use either Docker, or Singularity (if Docker or root privilege is not available). For deployment/serving, building either the CPU or the GPU image would suffice. If GPUs are not available, just go with the CPU image. For retraining/benchmarking, building the GPU image is required.

**Note**: only Docker is fully supported. The support for Singularity is partial. For Docker, it is recommended to add the user to the `docker` group first, so that only `docker run` is needed.

#### Using Docker

- Only option: build from local

```bash
(CPU) docker build -f Dockerfile_cpu -t ${ASKCOS_REGISTRY}/quarc:1.0-cpu .
(GPU) docker build -f Dockerfile_gpu -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu .
```

### Step 2/4: Download Trained Models

```bash
sh scripts/download_trained_models.sh
```

### Step 3/4: Start the Service

#### Using Docker

```bash
(CPU) sh scripts/serve_cpu_in_docker.sh
(GPU) sh scripts/serve_gpu_in_docker.sh
```

GPU-based container requires a CUDA-enabled GPU and the NVIDIA Container Toolkit. By default, the first GPU will be used.

Note that these scripts start the service in the background (i.e., in detached mode). So they would need to be explicitly stopped if no longer in use

```bash
docker stop quarc_service
```

### Step 4/4: Query the Service

- Sample query

```bash
curl http://0.0.0.0:9910/condition_prediction \
 --header "Content-Type: application/json" \
 --request POST \
 --data '{"smiles": ["CC(C)(C)OC(=O)O[C:1](=[O:2])[O:3][C:4]([CH3:5])([CH3:6])[CH3:7].[CH3:8][c:9]1[cH:10][c:11]([nH:12][cH:13]1)[CH:14]=[O:15]>CN(C)c1ccncc1.CC#N>[CH3:5][C:4]([CH3:6])([CH3:7])[O:3][C:1](=[O:2])[n:12]1[cH:13][c:9]([cH:10][c:11]1[CH:14]=[O:15])[CH3:8]"], "top_k": 3}'
```

- Sample response

```
[
  // one per reaction SMILES
  {
    "predictions": [
      {
        "rank": int,
        "agents": List[str],
        "temperature": str,
        "reactant_amounts": [
          {
            "reactant": str,
            "amount_range": str
          },
          ...
        ],
        "agent_amounts": [
          {
            "agent": str,
            "amount_range": str
          },
          ...
        ],
        "score": float
      },
     // ... up to top_k predictions per reaction SMILES
    ]
  },
  ...
]
```

### Unit Test for Serving (Optional)

Requirement: `requests` and `pytest` libraries (pip installable)

With the service started, run

```bash
pytest
```

## Retraining and benchmarking (GPU Required)

This section covers training the four-stage reaction condition recommendation pipeline:

1. **Stage 1 Agent Prediction**: Predicts chemical agents for a given reaction SMILES
2. **Stage 2 Temperature Prediction**: Predicts reaction temperature for a given reaction SMILES and agents
3. **Stage 3 Reactant Amount Prediction**: Predicts reactant amounts for a given reaction SMILES and agents
4. **Stage 4 Agent Amount Prediction**: Predicts agent amounts for a given reaction SMILES and agents

The four stages are trained indepdently, but at inference time, the predicted agents will serve as input to stage 2-4. The final predictions are made by enumerating and ranking the predictions from all four stages.

> [!Note]
> The full version of quarc relies on density values from Pistachio's proprietary web app to convert volume into molar amounts. Because these values are not publicly sharable, this open-source version uses a manually curated density file built from publicly avaliable sources (e.g., PubChem, NIST). We provide this density file for users wishing to preprocess and retrain models.
>
> While the pretrained open-source model was trained with Pistachio densities, users retraining from scratch with the open-source density file should expect slightly different behavior.

### Step 1/6: Environment Setup

Follow the instructions in Step 1/4 in the Serving section to build the GPU docker image. It should have the name `${ASKCOS_REGISTRY}/quarc:1.0-gpu`

Note: the Docker needs to be rebuilt before running whenever there is any change in code.

### Step 2/6: Data Preparation

Choose one of the following data preparation options:

- **Option 1: Pistachio Data**

If using Pistachio data (or data sharing Pistachio's format), the preprocessing pipeline can be applied directly by specifying the path to extracted Pistachio data.

Configure preprocessing settings in `./configs/preprocess_config.yaml`.

- **Option 2: Custom Data**

Skip the chunk_json and collect_dedup steps. Prepare your data in the `ReactionDatum` format:

```python
@dataclass
class AgentRecord:
    smiles: str
    amount: float | None

@dataclass
class ReactionDatum:
    document_id: str | None
    rxn_class: str | None
    date: str | None
    rxn_smiles: str

    reactants: list[AgentRecord]
    products: list[AgentRecord]
    agents: list[AgentRecord]

    temperature: float | None
```

Note: Amounts should be in moles for each reactant and agent. Once data is prepared in this format (including deduplication), the remaining preprocessing pipeline (vocabulary generation, filtering, document-level splitting) can be applied.

### Step 3/6: Preprocessing

Configure the environment variables to point to _absolute_ paths of your data files:

**For Pistachio Data:**

```bash
export RAW_DIR=/path/to/pistachio/extract

# Then run preprocessing
sh scripts/preprocess_in_docker_pistachio.sh
```

**For Custom Data:**

```bash
export FINAL_DEDUP_PATH=/path/to/final_deduped.pickle

# Then run preprocessing
sh scripts/preprocess_in_docker_custom.sh
```


### Step 4/6: Training

```bash
sh scripts/train_in_docker.sh
```

This trains all four stages independently. Progress and training logs are saved under `./logs`.

### Step 5/6: Model Selection and Pipeline Configuration

After training completes, you'll need to manually select the best models and configure the pipeline weights. This process involves two separate steps:

#### 1. Stage 1 Model Selection

Stage 1 (agent prediction) models are evaluated using greedy search on a subset of validation setduring training, but you may want to perform offline evaluation using beam search for final model selection.

```bash
sh scripts/stage1_offline_evaluation.sh
```

Review the results and select the best-performing checkpoint based on your accuracy requirements.

#### 2. Pipeline Weight Optimization

After selecting the best models for all 4 stages, optimize the weights used to combine scores from each stage:

```bash
sh scripts/optimize_weights_in_docker.sh
```

This performs hyperparameter tuning on the overall validation set to find optimal weights for ranking predictions across all stages.

#### 3. Configuration Update

Manually create `retrained_model_config.yaml` with:

- Selected model checkpoints for each stage
- Optimized pipeline weights
- Any stage-specific configuration parameters

Use `hybrid_pipeline_oss.yaml` as a template for the configuration format.

### Step 6/6: Prediction

Configure your selected models in `predict_in_docker.sh`, then run:

```bash
export PIPELINE_CONFIG_PATH="configs/best_model_config.yaml"
sh scripts/predict_in_docker.sh
```

This generates predictions for the test set and saves results to `./data/processed/overlap/overlap_test_predictions.json`. Adjust `--top-k` to change the number of predictions generated.

<!-- The estimated running times for benchmarking a typical dataset on a 32-core machine with 1 RTX3090 GPU are

- Preprocessing: ~20 mins
- Training: ~2 hours (4 stages)
- Testing: ~10 mins

The training parameters typically do not need to be adjusted, especially on larger datasets with more than 10,000 reactions. We leave it up to the user to adjust the training parameters in `scripts/train_in_docker.sh`, if you know what you are doing. -->

<!-- ## Converting Trained Model into Servable Archive (Optional)

If you want to create servable model archives from own checkpoints (e.g., trained on different datasets),
please refer to the archiving scripts (`scripts/archive_in_docker.sh`).
Change the arguments accordingly in the script before running.
It's mostly bookkeeping by replacing the data name and/or checkpoint paths; the script should be self-explanatory. Then execute the scripts with

```

sh scripts/archive_in_docker.sh

```

The servable model archive (.mar) will be generated under `./mars`. Serving newly archived models is straightforward; simply replace the `--models` args in `scripts/serve_{cpu,gpu}_in_{docker,singularity}.sh`

with the new model name and the .mar archive. The `--models` flag for torchserve can also take multiple arguments to serve multiple model archives concurrently. -->

## References

If you find our code or model useful, we kindly ask that you consider citing our work in your papers.

```bibtex
@article{Sun2025quarc,
  title={Data-Driven Recommendation of Agents, Temperature, and Equivalence Ratios for Organic Synthesis},
  author={Sun, Xiaoqi and Liu, Jiannan and Mahjour, Babak and Jensen, Klavs F and Coley, Connor W},
  journal={ChemRxiv},
  doi={10.26434/chemrxiv-2025-4wzkh},
  year={2025}
}
```
