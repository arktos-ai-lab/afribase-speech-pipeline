# AfriBase-Speech pipeline

Code for building AfriBase-Speech Tier 1a, a speech corpus for African languages made from US government lesson audio (FSI, Peace Corps, DLI), and for the Whisper-small experiments reported in the paper:

> Ernst van Gassen. AfriBase-Speech: Auditing, Tiering, and Packaging African Speech Resources from Government Public Domain Archives. DHASA 2026.

The pipeline goes from the course MP3s to fine-tuned Whisper-small checkpoints and FLEURS word error rates. It applies to any instruction recording with one dominant instruction language and public domain or permissively licensed content.

This repository holds code and result tables only. The corpus is on Hugging Face as [vGassen/afribase-speech](https://huggingface.co/datasets/vGassen/afribase-speech) (DOI [10.57967/hf/11727](https://doi.org/10.57967/hf/11727)), together with the VOA broadcast audio of the companion paper as Tier 1b. The five fine-tuned Whisper-small models are at [vGassen/afribase-speech-whisper-small](https://huggingface.co/vGassen/afribase-speech-whisper-small). Both are grouped in the [AfriBase-Speech collection](https://huggingface.co/collections/vGassen/afribase-speech-6b001c6a16e93128db1f5e5c).

## Stages

| Stage | Script | Hardware |
|---|---|---|
| Download course MP3s from the Internet Archive | `download_tier1.py` | network |
| Convert to 16 kHz mono, split at pauses (pydub, at least 500 ms below -35 dBFS), keep segments of 1 to 15 s with SNR of at least 10 dB and speech fraction of at least 0.60 | `segment_lessons.py` | CPU |
| Optional stricter re-filtering of segment manifests | `quality_filter.py` | CPU |
| Pseudo-label each segment with Whisper large-v3, language set in advance; package as a Hugging Face dataset with a 95/5 train/validation split | `package_corpus.py` | GPU |
| Remove English instructor speech with Whisper large-v3 language identification (faster-whisper) | `langid_filter.py` | GPU |
| Fine-tune Whisper-small per language: 10 epochs, batch size 8, learning rate 1e-5, best epoch by validation WER | `finetune_whisper.py` | GPU |
| Evaluate on the FLEURS test split and print the WER table | `eval_fleurs.py` | GPU |

`run_pipeline.sh` runs all stages for Amharic, Hausa, Lingala, Swahili and Yoruba:

```bash
pip install -r requirements.txt   # plus a CUDA build of torch for your driver
bash run_pipeline.sh              # or: bash run_pipeline.sh lin swh
```

Every stage writes its progress to disk as it goes and skips finished work on restart, so after a crash or interruption you rerun the same command. The paper's runs used one laptop GPU (NVIDIA RTX 5090 Laptop, 24 GB); fine-tuning took between 16 minutes (Hausa) and 3.1 hours (Amharic) per language.

The segmenter runs with `--skip-langid`, so English instructor speech stays in the segments until `langid_filter.py` removes it. A segment is dropped when its most likely language is English. The detected language is not required to match the target, because detection is unreliable on short segments of low-resource languages.

## Results in this repository

| File | Contents |
|---|---|
| `results/fleurs_10ep_v2.csv` | Whisper-small zero-shot, and fine-tuned on segments before English removal |
| `results/fleurs_10ep_v3.csv` | Whisper-small fine-tuned on segments after English removal |
| `results/final_table_10ep_v3.txt` | WER table printed from the v3 file |
| `results/langid_v3/<lang>.jsonl` | Per-segment language identification output: top language and its probability, English and target probability |

WER and CER are stored as fractions (1.0 = 100%).

## Licences

The code is MIT licensed (see `LICENSE`). That licence covers the software only. The source recordings are works of the US federal government and are in the public domain in the US under 17 U.S.C. 105. The compiled dataset (segmentation, filtering, labels, metadata, packaging) is released under CC BY 4.0. FLEURS, used for evaluation, is CC BY 4.0. Whisper models are MIT licensed.

## Citation

```bibtex
@inproceedings{vangassen2026afribasespeech,
  author    = {van Gassen, Ernst},
  title     = {{AfriBase-Speech}: Auditing, Tiering, and Packaging African Speech Resources from Government Public Domain Archives},
  booktitle = {Proceedings of the Digital Humanities Association of Southern Africa Conference (DHASA 2026)},
  year      = {2026}
}
```

Ernst van Gassen, Independent researcher at Arktos AI Lab. ORCID [0009-0000-0314-6441](https://orcid.org/0009-0000-0314-6441).
