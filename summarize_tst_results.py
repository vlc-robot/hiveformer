
import os
import numpy as np
import jsonlines
import collections
import tap


class Arguments(tap.Tap):
    result_file: str


def main(args):
    results = collections.defaultdict(list)
    with jsonlines.open(args.result_file, 'r') as f:
        for item in f:
            results[item['checkpoint']].append((item['task'], item['sr']))

    for ckpt, res in results.items():
        print('\n', ckpt)
        print(','.join([x[0] for x in res]))

        print(','.join(['%.2f' % (x[1]*100) for x in res]))

        print(np.mean([x[1] for x in res]) * 100)


if __name__ == '__main__':
    args = Arguments().parse_args()
    main(args)
