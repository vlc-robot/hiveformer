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
            results[item['checkpoint']].append(item['sr'])

    avg_results = []
    for k, v in results.items():
        print(k, len(v), np.mean(v))
        avg_results.append((k, np.mean(v)))

    print()
    print('Best checkpoint and SR')
    avg_results.sort(key=lambda x: -x[1])
    print(avg_results[0])


if __name__ == '__main__':
    args = Arguments().parse_args()
    main(args)
