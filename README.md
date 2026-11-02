# SeqGraphLLM

SeqGraphLLM explores program graph representations for large language models on vulnerability detection and Python type inference.

## Workflow

1. **Prepare data.** Obtain the task datasets: Big-Vul or PrimeVul for vulnerability detection, and ManyTypes4Py for type inference. The full datasets are not included in this repository.
2. **Generate program graphs.** Use Joern to build code property graphs and extract abstract syntax, control flow, and data dependency graphs. The processing scripts are in [`code/data_processing/`](code/data_processing/).
3. **Build graph representations.** Convert graphs into DOT, textual, or code-style representations. Optionally apply degree-based compression (DBC), degree-based pruning (DBP), and graph fitting to the input token budget.
4. **Configure the model and prompts.** Prepare a model API endpoint and set the model path, context window, prompts, and examples. Vulnerability detection configurations are in [`code/vulnerability_detection/`](code/vulnerability_detection/). Replace machine-specific model paths before running.
5. **Run inference.** Combine source code with the selected graph representation and send it to the model. Use the code-only setting as a baseline. Predictions are saved to SQLite databases for subsequent analysis.

## Main Entry Points

| Task | Script |
| --- | --- |
| Big-Vul / PrimeVul graph preparation | [`regenerate_bigvul.py`](code/data_processing/regenerate_bigvul.py) |
| ManyTypes4Py graph preparation | [`regenerate_manytypes4py.py`](code/data_processing/regenerate_manytypes4py.py) |
| Vulnerability detection | [`vulnerability_detection.py`](code/vulnerability_detection/vulnerability_detection.py) |
| Python type inference | [`type_inference.py`](code/type_inference/type_inference.py) |
| Natural-language explanation generation | [`gen_nl_explanations.py`](code/scripts/gen_nl_explanations.py) |

See [`requirements.txt`](requirements.txt) for Python dependencies. Joern, the task datasets, and a model service must be prepared separately. Model-specific token counting also requires `transformers`.

## License

See [LICENSE](LICENSE).
