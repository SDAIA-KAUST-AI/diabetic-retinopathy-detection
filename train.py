import datetime
from argparse import ArgumentParser

import torch

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelSummary

from src.trainer import ViTLightningModule


def main():
    """ Neural network trainer entry point. """

    parser = ArgumentParser(description='KAUST-SDAIA Diabetic Retinopathy')
    parser.add_argument('--tag', action='store', type=str,
                        help='Extra suffix to put on the artefact dir name')
    parser.add_argument('--debug', action='store_true',
                        help="Dummy training cycle for testing purposes")
    parser.add_argument('--convert-checkpoint', action='store', type=str,
                        help='Convert a checkpoint from training to pickle-independent '
                             'predictor-compatible directory')

    args = parser.parse_args()

    torch.set_float32_matmul_precision('high') # for V100/A100

    if args.convert_checkpoint is not None:

        print("Converting checkpoint", args.convert_checkpoint)

        checkpoint = torch.load(args.convert_checkpoint, map_location="cpu")
        print(list(checkpoint.keys()))

        model = ViTLightningModule.load_from_checkpoint(
            args.convert_checkpoint,
            map_location="cpu",
            hparams_file="tmp_ckpt_deleteme.yaml")

        model.save_checkpoint_dk("tmp_checkp_path_deleteme")

        print("Saved checkpoint. Done.")

    else:

        print("Start training")

        fast_dev_run = True if args.debug == True else False

        model = ViTLightningModule(fast_dev_run)

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        art_dir_name = (f"{datetime_str}" +
                        (f"_{args.tag}" if args.tag is not None else ""))
        logger = TensorBoardLogger(save_dir=".", name="lightning_logs", version=art_dir_name)

        trainer = Trainer(
            logger=logger,
            benchmark=True,
            devices="auto",
            accelerator="auto",
            max_epochs=-1,
            callbacks=[
                ModelSummary(max_depth=-1),
                ],
            fast_dev_run=fast_dev_run,
            log_every_n_steps=10,
            )

        trainer.fit(
            model,
            train_dataloaders=model._train_dataloader,
            val_dataloaders=model._val_dataloader,
            )

        print("Training done")


if __name__ == "__main__":
    main()
