import json
from data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset
import glob
import six
from keras.callbacks import Callback
import argparse
import sys


def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    all_checkpoint_files = [ ff.replace(".index" , "" ) for ff in all_checkpoint_files ] # to make it work for newer versions of keras
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))
    print(latest_epoch_checkpoint)
    return latest_epoch_checkpoint


def masked_categorical_crossentropy(gt, pr):
    from keras.losses import categorical_crossentropy
    mask = 1 - gt[:, :, 0]
    return categorical_crossentropy(gt, pr) * mask


class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "." + str(epoch))
            print("saved ", self.checkpoints_path + "." + str(epoch))


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          val_steps_per_epoch=512,
          gen_use_multiprocessing=False,
          ignore_zero_class=False,
          optimizer_name='adam',
          do_augment=False,
          augmentation_name="aug_all"):
    print("I AM HERE")
    from models.all_models import model_from_name
    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None

    if optimizer_name is not None:

        if ignore_zero_class:
            loss_k = masked_categorical_crossentropy
        else:
            loss_k = 'categorical_crossentropy'

        model.compile(loss=loss_k,
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying training dataset")
        print("I AM HERE")
        verified = verify_segmentation_dataset(train_images,
                                               train_annotations,
                                               n_classes)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images,
                                                   val_annotations,
                                                   n_classes)
            assert verified

    train_gen = image_segmentation_generator(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width,
        do_augment=do_augment, augmentation_name=augmentation_name)

    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)

    callbacks = [
        CheckpointsCallback(checkpoints_path)
    ]

    if not validate:
        model.fit_generator(train_gen, steps_per_epoch,
                            epochs=epochs, callbacks=callbacks)
    else:
        model.fit_generator(train_gen,
                            steps_per_epoch,
                            validation_data=val_gen,
                            validation_steps=val_steps_per_epoch,
                            epochs=epochs, callbacks=callbacks,
                            use_multiprocessing=gen_use_multiprocessing)

def train_action(parser):
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

    parser.add_argument("--steps_per_epoch", type=int, default=1430)
    parser.add_argument("--optimizer_name", type=str, default="adam")

    args = parser.parse_args()
    print(args)
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

if __name__ == "__main__":
    main_parser = argparse.ArgumentParser()
    train_action(main_parser)
