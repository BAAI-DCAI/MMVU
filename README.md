<h1 align = "center">
  MMVU: Unveiling MLLM Misresponses Despite Visual Understanding
</h1>

<p align="center">
    <a href="http://arxiv.org/abs/2406.10638">
        <img alt="Paper" src="http://img.shields.io/badge/Paper-arXiv%3A2406.10638-B31B1B.svg">
    </a>
    <a href="https://huggingface.co/datasets/BAAI/Multimodal-Robustness-Benchmark">
        <img alt="Dataset" src="https://img.shields.io/badge/🤗%20Dataset-MMR%20Benchmark-yellow">
    </a>
</p>
<p align="center">
    <a href="https://huggingface.co/AI4VR/Bunny-MMR-3B">
        <img alt="Model Bunny-MMR-3B" src="https://img.shields.io/badge/🤗%20Model-Bunny--MMR--3B-green">
    </a>
    <a href="https://huggingface.co/AI4VR/Bunny-MMR-4B">
        <img alt="Model Bunny-MMR-4B" src="https://img.shields.io/badge/🤗%20Model-Bunny--MMR--4B-green">
    </a>
    <a href="https://huggingface.co/AI4VR/Bunny-MMR-8B">
        <img alt="Model Bunny-MMR-8B" src="https://img.shields.io/badge/🤗%20Model-Bunny--MMR--8B-green">
    </a>
</p>


This repo contains the official evaluation code and dataset for the paper“Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly”.

## 📢 News and Updates
* 2025.02.27 🎉 The paper appears on CVPR25'!
* 2024.06.18 🔥 **Checkpoints are released!** Check more details in HuggingFace: [Bunny-MMR-3B](https://huggingface.co/AI4VR/Bunny-MMR-3B), [Bunny-MMR-4B](https://huggingface.co/AI4VR/Bunny-MMR-4B), [Bunny-MMR-8B](https://huggingface.co/AI4VR/Bunny-MMR-8B).
* 2024.06.18 🔥 **Paper is ready.** Check more details in [arXiv](https://arxiv.org/abs/2406.10638).
* 2024.06.17 🔥 **Demo is available.** Check more details in [link](https://law1223.github.io/Multimodal-Robustness-Benchmark/). Welcome everyone to try it!
* 2024.06.13 🔥 **MMR benchmark and MMR-data are released!** Check more details in [HuggingFace](https://huggingface.co/datasets/BAAI/Multimodal-Robustness-Benchmark).

## 🙌 How to Add a New Model to MMR Benchmark

* If your model is on HuggingFace, please create an Issue to provide the following information:
  - Model ID
  - Chat template
  - Preferred sampling parameters

We will then add the necessary script to our repository and handle the inference and evaluation for you.

* If you prefer to run inference on your model yourself or create a PR to add your model to our repository, please follow the instructions in the [evaluation](https://github.com/BAAI-DCAI/Multimodal-Robustness-Benchmark/tree/main/evaluation) folder.


## 📇 Contents
- [MMR-benchmark](#%EF%B8%8F-mmr-benchmark)
- [Evaluation](#-evaluation)
- [Leaderboard](#-leaderboard)
- [MMR-data](#-mmr-data)
- [Training](#-training)
- [Quickstart](#-quickstart)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgement](#-acknowledgement)

## ⚖ MMR-benchmark

Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in visual understanding and reasoning, providing reasonably accurate answers, such as image descriptions. This has spurred extensive research into evaluating MLLMs. Most evaluation benchmarks assume that incorrect answers indicate a lack of understanding of the visual content. However, our findings reveal that, in many cases, MLLMs answer questions incorrectly despite correctly understanding the visual content. This suggests that incorrect answers do not necessarily imply a lack of comprehension but may instead result from a lack of robustness to leading questions.

Multimodal Large Language Models (MLLMs) have displayed remarkable performance in multi-modal tasks, particularly in visual comprehension. However, we reveal that MLLMs often generate incorrect answers even when they understand the visual content. To this end, we manually construct a benchmark with 12 categories and design evaluation metrics that assess the degree of error in MLLM responses even when the visual content is seemingly understood. Based on this benchmark, we test 15 leading MLLMs and analyze the distribution of attention maps and logits of some MLLMs. 

<p align="center">
  <img src="./figure/cover_fig.png" alt="Logo">
</p>

To explore this phenomenon, we introduce the MLLM Misresponses despite Visual Understanding (MMVU) benchmark. The MMVU dataset consists of a benchmarking dataset for evaluating models as well as a training dataset. The former is curated by human annotators together with the appropriate metrics and analysis on the MLLM's attention and logit behavior. Based on the experiments, we propose a data construction pipeline to build a training dataset and prompting strategies to enhance the accuracy of MLLM responses.challenge its resistance to interference.

![mmr_benchmark](./figure/mmr_benchmark.png)


## 🏁 Evaluation

Please refer to our [evaluation](https://github.com/BAAI-DCAI/Multimodal-Robustness-Benchmark/tree/dev/evaluation) folder for more details.

## 🏆 Leaderboard

![table01](./figure/table01.png) 

![table02](./figure/table02.png) 

## 🚩 MMR-data

To enhance MLLMs' understanding capability and robustness, we propose a data construction method using GPT-4V to generate paired positive and negative samples for instruction tuning. The method includes three steps: 1) Information extraction. We implicitly and comprehensively extract detailed information from images, including text, object attributes, human characteristics, relationships between objects, relationships between people, events, and overall perception. 2) Instruction tuning data generation. We generate positive samples using the extracted information and construct negative samples that directly contradict the positive ones. 3) Sample filtering. We filter samples through keyword matching to remove those with uncertain answers and redundant phrases.

![data_collection](./figure/data_collection.png)


### Data generation
- Generate conversations based on GPT-4V
  
```shell
python dataset/data_generation.py \
      --input_file /path/to/input.json \
      --output_file /path/to/output.json \
      --image_folder /path/to/image folder \
      --api_key api_key
```

- Reformat the JSON

```shell
python dataset/data_reformat.py \
      --input /path/to/input.json \
      --output_pos /path/to/output_pos.json \
      --output_neg /path/to/output_neg.json \
      --output_merge /path/to/merged_output.json
```

- Filter the JSON (Optional)

```shell
python dataset/data_filtering.py \
      --input /path/to/input.json \
      --output /path/to/output.json
```

## 🤖 Training

- We build the model based on [Bunny](https://github.com/BAAI-DCAI/Bunny). Please refer to [Bunny](https://github.com/BAAI-DCAI/Bunny) for more details.
- Training details and checkpoints.
  
| Checkpoint                                                   | Vision Encoder                                               | LLM                                                          | Pretrain lr | Pretrain weights                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | :---------: | ------------------------------------------------------------ |
| [Bunny-MMR-3B](https://huggingface.co/AI4VR/Bunny-MMR-3B) | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)    |    5e-4     | [bunny-pretrain-phi-2-siglip](https://huggingface.co/BAAI/bunny-pretrain-phi-2-siglip) |
| [Bunny-MMR-4B](https://huggingface.co/AI4VR/Bunny-MMR-4B) | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |    1e-3     | [bunny-pretrain-phi-3-siglip](https://huggingface.co/BoyaWu10/bunny-pretrain-phi-3-siglip) |
| [Bunny-MMR-8B](https://huggingface.co/AI4VR/Bunny-MMR-8B) | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |    1e-3     | [bunny-pretrain-llama3-8b-siglip](https://huggingface.co/BoyaWu10/bunny-pretrain-llama3-8b-siglip) |

## 🌟 Quickstart

Here we show a code snippet to show you how to use the model with transformers.

Before running the snippet, you need to install the following dependencies:

```shell
pip install torch transformers accelerate pillow
```

```python
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
torch.set_default_device('cpu')  # or 'cuda'

offset_bos = 1 # for Bunny-MMR-8B and AI4VR/Bunny-MMR-4B
# offset_bos = 0 for Bunny-MMR-3B

# create model
model = AutoModelForCausalLM.from_pretrained(
    'AI4VR/Bunny-MMR-8B', # or 'AI4VR/Bunny-MMR-3B' or 'AI4VR/Bunny-MMR-4B'.
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    'AI4VR/Bunny-MMR-8B', # or 'AI4VR/Bunny-MMR-3B' or 'AI4VR/Bunny-MMR-4B'.
    trust_remote_code=True)

# text prompt
prompt = 'text prompt'
text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0)

# image input
image = Image.open('path/to/image')
image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)

# generate
output_ids = model.generate(
    input_ids,
    images=image_tensor,
    max_new_tokens=100,
    use_cache=True)[0]

print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())
```

## 🔗 Citation
If you find this repository helpful, please cite the paper below.

```bibtex
@misc{liu2024seeing,
    title={Seeing Clearly, Answering Incorrectly: A Multimodal Robustness Benchmark for Evaluating MLLMs on Leading Questions},
    author={Yexin Liu and Zhengyang Liang and Yueze Wang and Muyang He and Jian Li and Bo Zhao},
    year={2024},
    eprint={2406.10638},
    archivePrefix={arXiv},
}
```

## 🧾 License
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](./LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC--BY--4.0-orange.svg)](./LICENSE)
[![Weight License](https://img.shields.io/badge/Weight%20License-CC--BY--4.0-red.svg)](./LICENSE)

The project employs specific datasets and checkpoints that are governed by their original licenses. Users must adhere to all terms and conditions outlined in these licenses. The checkpoints are restricted to uses that comply with the license agreements of Bunny, LLaMA 3, Phi-2, Phi-3, and GPT-4. The dataset is provided under the CC-BY-4.0 license.


## 📫 Acknowledgement

- The training of this work is built upon the [Bunny: A family of lightweight multimodal models](https://github.com/BAAI-DCAI/Bunny).
- This work utilizes LLMs from [Phi-2](https://huggingface.co/microsoft/phi-2), [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), and [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
