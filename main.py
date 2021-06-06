from belabot.engine.belot import Belot
from belabot.engine.player import AiPlayer, BigBrain, RandomPlayer
from belabot.engine.util import get_logger
import time
import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-checkpoint', type=argparse.FileType('rb'), dest='checkpoint')
    parser.add_argument('--save-as', type=str, default=None, dest='model_filename')
    parser.add_argument('--no-save', action='store_false', dest='do_save')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--test', action='store_false', dest='train')
    return parser.parse_args()

def main(args):
    log = get_logger(__name__)
    big_brain = BigBrain(args.checkpoint)
    if args.train:
        players = [
            AiPlayer("0", big_brain),
            AiPlayer("1", big_brain),#RandomPlayer("1"),
            AiPlayer("2", big_brain),
            AiPlayer("3", big_brain),#RandomPlayer("3"),
        ]
        grad = torch.enable_grad()
    else:
        players = [
            AiPlayer("0", big_brain),
            RandomPlayer("1"),
            AiPlayer("2", big_brain),
            RandomPlayer("3")
        ]
        grad = torch.no_grad()
    #log.info("Grad: {}".format(grad))
    with grad:
        belot = Belot(players, do_train=args.train)
        #belot.play()
        for i in range(args.epochs):
            metrics = belot.round(i % 4)
            metrics.type = 'Train' if args.train else 'Test'
            log.info(metrics)
        if args.do_save:
            big_brain.save(args.model_filename)

if __name__ == '__main__':
    args = get_args()
    main(args)
