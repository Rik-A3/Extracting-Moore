# Extracting-Moore

Code for reproduction of the experiments of Extracting Moore Machines from Transformers using Queries and Counterexamples (IDA 2025).

# Install
These experiments use Python 3.12 To install run:

```shell
bash setup.sh
```

# Experiments
All experiments can be found in `experiments`. The hyperparameters for each experiment can be found in their respective yaml file in `configs/`.
To run an experiment, use the following command:
```shell
python run_experiments.py <config_path>
```

For example, to train and extract a transformer with by training on only positive examples run:
```shell
python run_experiments.py configs/positive_only.yaml
```
By default, training metrics and results are logged to wandb.