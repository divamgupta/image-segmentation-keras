#!/usr/bin/env python

import sys
import argparse

from .train import train
from .predict import predict, predict_multiple, predict_video, evaluate
from .data_utils.data_loader import verify_segmentation_dataset
from .data_utils.visualize_dataset import visualize_segmentation_dataset


def train_action(command_parser):
    parser = command_parser.add_parser('train')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_images", type=str, required=True)
    parser.add_argument("--train_annotations", type=str, required=True)

    parser.add_argument("--n_classes", type=int, required=True)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)

    parser.add_argument('--not_verify_dataset', action='store_false')
    parser.add_argument("--checkpoints_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument('--validate', action='store_true')
    parser.add_argument("--val_images", type=str, default="")
    parser.add_argument("--val_annotations", type=str, default="")

    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--load_weights", type=str, default=None)
    parser.add_argument('--auto_resume_checkpoint', action='store_true')

    parser.add_argument("--steps_per_epoch", type=int, default=512)
    parser.add_argument("--optimizer_name", type=str, default="adam")

    def action(args):
        return train(model=args.model_name,
                     train_images=args.train_images,
                     train_annotations=args.train_annotations,
                     input_height=args.input_height,
                     input_width=args.input_width,
                     n_classes=args.n_classes,
                     verify_dataset=args.not_verify_dataset,
                     checkpoints_path=args.checkpoints_path,
                     epochs=args.epochs,
                     batch_size=args.batch_size,
                     validate=args.validate,
                     val_images=args.val_images,
                     val_annotations=args.val_annotations,
                     val_batch_size=args.val_batch_size,
                     auto_resume_checkpoint=args.auto_resume_checkpoint,
                     load_weights=args.load_weights,
                     steps_per_epoch=args.steps_per_epoch,
                     optimizer_name=args.optimizer_name)

    parser.set_defaults(func=action)


def predict_action(command_parser):

    parser = command_parser.add_parser('predict')
    parser.add_argument("--checkpoints_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, default="", required=True)
    parser.add_argument("--output_path", type=str, default="", required=True)

    def action(args):
        input_path_extension = args.input_path.split('.')[-1]
        if input_path_extension in ['jpg', 'jpeg', 'png']:
            return predict(inp=args.input_path, out_fname=args.output_path,
                           checkpoints_path=args.checkpoints_path)
        else:
            return predict_multiple(inp_dir=args.input_path,
                                    out_dir=args.output_path,
                                    checkpoints_path=args.checkpoints_path)

    parser.set_defaults(func=action)


def predict_video_action(command_parser):
    parser = command_parser.add_parser('predict_video')
    parser.add_argument("--input", type=str, default=0, required=False)
    parser.add_argument("--output_file", type=str, default="", required=False)
    parser.add_argument("--checkpoints_path", required=True)
    parser.add_argument("--display", action='store_true', required=False)

    def action(args):
        return predict_video(inp=args.input,
                             output=args.output_file,
                             checkpoints_path=args.checkpoints_path,
                             display=args.display,
                             )

    parser.set_defaults(func=action)


def evaluate_model_action(command_parser):

    parser = command_parser.add_parser('evaluate_model')
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--segs_path", type=str, required=True)
    parser.add_argument("--checkpoints_path", type=str, required=True)

    def action(args):
        print(evaluate(
            inp_images_dir=args.images_path, annotations_dir=args.segs_path,
            checkpoints_path=args.checkpoints_path))

    parser.set_defaults(func=action)


def verify_dataset_action(command_parser):

    parser = command_parser.add_parser('verify_dataset')
    parser.add_argument("--images_path", type=str)
    parser.add_argument("--segs_path", type=str)
    parser.add_argument("--n_classes", type=int)

    def action(args):
        verify_segmentation_dataset(
            args.images_path, args.segs_path, args.n_classes)

    parser.set_defaults(func=action)


def visualize_dataset_action(command_parser):

    parser = command_parser.add_parser('visualize_dataset')
    parser.add_argument("--images_path", type=str)
    parser.add_argument("--segs_path", type=str)
    parser.add_argument("--n_classes", type=int)
    parser.add_argument('--do_augment', action='store_true')

    def action(args):
        visualize_segmentation_dataset(args.images_path, args.segs_path,
                                       args.n_classes,
                                       do_augment=args.do_augment)

    parser.set_defaults(func=action)


def main():
    assert len(sys.argv) >= 2, \
        "python -m keras_segmentation <command> <arguments>"

    main_parser = argparse.ArgumentParser()
    command_parser = main_parser.add_subparsers()

    # Add individual commands
    train_action(command_parser)
    predict_action(command_parser)
    predict_video_action(command_parser)
    verify_dataset_action(command_parser)
    visualize_dataset_action(command_parser)
    evaluate_model_action(command_parser)

    args = main_parser.parse_args()

    args.func(args)
