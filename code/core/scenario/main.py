import simulator

def main(args):
    # create trainer
    trainer = simulator.trainer.create(args)
    # training
    trainer.load()
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.test()
    else:
        raise NotImplementedError
