from belabot.engine.belot import Belot
from belabot.engine.player import AiPlayer, BigBrain, RandomPlayer
import time
import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-checkpoint', type=argparse.FileType('rb'), dest='checkpoint')
    parser.add_argument('--save-as', type=str, default=None, dest='model_filename')
    parser.add_argument('--no-save', action='store_false', dest='do_save')
    parser.add_argument('--epochs', type=int, default=100)
    return parser.parse_args()

def main(args):
    big_brain = BigBrain(args.checkpoint)
    big_brain.model.eval()
    print(list(big_brain.model.children()))
    players = [
        AiPlayer("0", big_brain),
        RandomPlayer("1"),
        AiPlayer("2", big_brain),
        RandomPlayer("3"),
    ]
    belot = Belot(players)
    #belot.play()
    since = time.time()
    for i in range(args.epochs):
        print(i, end='\t')
        belot.round(i % 4)
        new_time = time.time()
        #log.debug(new_time - since)
        since = new_time
    if args.do_save:
        big_brain.save(args.model_filename)

if __name__ == '__main__':
    args = get_args()
    main(args)
