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

```json
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

<!-- ## Retraining and benchmarking (GPU Required)

The full version of quarc relies on density values from Pistachio's proprietary web app to convert volume into molar amounts. Since these values cannot be shared, this open-source version uses a manually curated density file built from publicly avaliable sources (e.g., PubChem, NIST). We provide this density file for users wishing to preprocess and retrain models.

> [!Note]
> While the pretrained open-source model was originally trained with Pistachio-provided densities, users retraining from scratch should expect slightly different behavior when using the open-source densities we supply.

### Step 1/4: Environment Setup

Follow the instructions in Step 1/4 in the Serving section to build the GPU docker image. It should have the name `${QUARC_REGISTRY}/quarc:1.0-gpu`

Note: the Docker needs to be rebuilt before running whenever there is any change in code.

### Step 2/4: Data Preparation

Note that the preprocessing stage requires open-source reaction classification (no proprietary NameRxn needed). Various reaction classification tools are available in the community including RDKit reaction fingerprints.

- Option 1: provide pre-split reaction data

Prepare the raw .csv files for train, validation and test. The required columns are "rxn_smiles" and "conditions" (containing agent SMILES, temperature, and amounts). This is the typical setting, where the pre-split files are supplied.

You can also include other columns in the .csv files, which will all be saved during preprocessing (in `reactions.processed.json.gz`).

- Option 2: provide unsplit reaction data

It is also possible to supply a single .csv file containing all reactions and let the preprocessing engine handle the splitting. In this case, by default, reactions with failed condition extraction will be filtered out, after which the remaining reactions will be deduplicated, split into train/val/test splits.

### Step 3/4: Path Configuration

- Case 1: if pre-split reaction data is provided

Configure the environment variables in `./scripts/benchmark_in_docker_presplit.sh`, especially the paths, to point to the _absolute_ paths of raw files and desired output paths.

```

# benchmark_in_docker_presplit.sh

...
export DATA_NAME="my_new_reactions"
export TRAIN_FILE=$PWD/new_data/raw_train.csv
export VAL_FILE=$PWD/new_data/raw_val.csv
export TEST_FILE=$PWD/new_data/raw_test.csv
...

```

- Case 2: if unsplit reaction data is provided

Configure the environment variables in `./scripts/benchmark_in_docker_unsplit.sh`, especially the paths, to point to the _absolute_ paths of the raw file and desired output paths.

```

# benchmark_in_docker_unsplit.sh

...
export DATA_NAME="my_new_reactions"
export ALL_REACTION_FILE=$PWD/new_data/all_reactions.csv
...

```

the default train/val/test ratio is 98:1:1, which can be adjusted too. For example, if you want to use most of the data for training and very little for validation or testing,

```

# benchmark_in_docker_unsplit.sh

...
bash scripts/preprocess_in_docker.sh --split_ratio 99:1:0
bash scripts/train_in_docker.sh
bash scripts/predict_in_docker.sh

```

### Step 4/4: Training and Benchmarking

Run benchmarking on a machine with GPU using

```

sh scripts/benchmark_in_docker_presplit.sh

```

for pre-split data, or

```

sh scripts/benchmark_in_docker_unsplit.sh

```

for unsplit data. This will run the preprocessing, training and predicting for the QUARC model with top-n accuracies up to n=10 as the final outputs. Progress and result logs will be saved under `./logs`.

The estimated running times for benchmarking a typical dataset on a 32-core machine with 1 RTX3090 GPU are

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
