#!/bin/bash
# End-to-end AfriBase-Speech Tier 1a pipeline for the five evaluation languages:
#   download -> segment + quality filter -> pseudo-label + package -> LangID filter
#   -> fine-tune Whisper-small -> FLEURS eval -> table.
# Every stage is resumable: rerunning after a crash continues where it stopped.
#   download_tier1.py     skips files already in source_audio/<lang>/
#   segment_lessons.py    skips source files already listed in segments/<lang>.jsonl
#   package_corpus.py     fsynced per-batch pseudo-label checkpoint; --resume skips built languages
#   langid_filter.py      per-segment fsynced JSONL in results/langid_v3/
#   finetune_whisper.py   resumes from the last per-epoch checkpoint
#   eval_fleurs.py        per-utterance fsynced JSONL in results/ckpt/
# Usage: bash run_pipeline.sh [lang ...]     (default: amh hau lin swh yor)
set -e
cd "$(dirname "$0")"
LANGS="${*:-amh hau lin swh yor}"
mkdir -p logs

echo "$(date) stage 0: download" | tee -a logs/pipeline.txt
python download_tier1.py --lang $LANGS --out source_audio >> logs/download.txt 2>&1

echo "$(date) stage 1: segment + quality filter (CPU)" | tee -a logs/pipeline.txt
for lang in $LANGS; do
  python segment_lessons.py --lang $lang --input source_audio/$lang/ --out segments/ \
      --source-name FSI --skip-langid >> logs/segment.txt 2>&1
done

echo "$(date) stage 2: pseudo-label with Whisper large-v3 + package (GPU)" | tee -a logs/pipeline.txt
python package_corpus.py --segments-dir segments/ --audio-root segments/ \
    --out corpus_v2/ --device cuda --resume >> logs/package.txt 2>&1

echo "$(date) stage 3: LangID filter, drop English instructor speech (GPU)" | tee -a logs/pipeline.txt
python langid_filter.py --lang $LANGS --in-root corpus_v2 --out-root corpus_v3 >> logs/langid.txt 2>&1

echo "$(date) stage 4: fine-tune Whisper-small (GPU)" | tee -a logs/pipeline.txt
python finetune_whisper.py --corpus-root corpus_v3/ --out-root runs/ \
    --device cuda --epochs 10 --no-fleurs-eval >> logs/finetune.txt 2>&1

echo "$(date) stage 5: FLEURS eval" | tee -a logs/pipeline.txt
for lang in $LANGS; do
  python eval_fleurs.py --model runs/whisper-small-${lang}/best/ --lang ${lang} \
      --label whisper_small_ft_${lang} --out results/fleurs.csv >> logs/eval.txt 2>&1
done
python eval_fleurs.py --model openai/whisper-small --lang $LANGS \
    --label whisper_small_zs --out results/fleurs.csv >> logs/eval.txt 2>&1
python eval_fleurs.py --table results/fleurs.csv > results/final_table.txt 2>&1
echo "$(date) done" | tee -a logs/pipeline.txt
