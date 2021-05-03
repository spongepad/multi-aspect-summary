import argparse
from models.bart import BART
from transformers.models.bart import BartForConditionalGeneration
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None, type=str)
parser.add_argument("--model_binary", default=None, type=str)
parser.add_argument("--output_dir", default='logs/aspect_summary', type=str)
args = parser.parse_args()

with open(args.hparams) as f:
    hparams = yaml.load(f)
    
inf = BART.load_from_checkpoint(hparams=hparams, checkpoint_path=args.model_binary)

inf._model.save_pretrained(args.output_dir)