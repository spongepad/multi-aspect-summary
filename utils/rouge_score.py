import argparse
from rouge import Rouge

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", "--src",
    type=str,
    default=None,
    help='src',
)
parser.add_argument(
    "-t", "--tar",
    type=str,
    default=None,
    help='tar',
)

args = parser.parse_args()

with open(args.src, encoding='utf-8-sig') as f:
    srcs = []
    for line in f:
        if(line[-1] == '\n'):
            line = line[:-1]
        srcs.append(line)

with open(args.tar, encoding='utf-8-sig') as f:
    tars = []
    for line in f:
        if(line[-1] == '\n'):
            line = line[:-1]
        tars.append(line)

rouge = Rouge()
scores = rouge.get_scores(srcs, tars, avg=True)

print(scores)
