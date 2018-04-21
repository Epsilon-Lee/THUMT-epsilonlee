#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import six
import sys

sys.path.append('/home/guanlin/Workspace/Repos/THUMT-dev/')
from tensorflow.python import debug as tf_debug

import numpy as np
import tensorflow as tf
import thumt.data.cache as cache
import thumt.data.dataset as dataset
import thumt.data.record as record
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.hooks as hooks
import thumt.utils.inference as inference
import thumt.utils.optimize as optimize
import thumt.utils.parallel as parallel


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training neural machine translation models",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, nargs=2,
                        help="Path of source and target corpus")
    parser.add_argument("--record", type=str,
                        help="Path to tf.Record data")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to saved models")
    parser.add_argument("--vocabulary", type=str, nargs=2,
                        help="Path of source and target vocabulary")
    parser.add_argument("--validation", type=str,
                        help="Path of validation file")
    parser.add_argument("--references", type=str, nargs="+",
                        help="Path of reference files")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint")

    # train mode, model and configuration
    parser.add_argument("--train_mode", type=str, required=True,
                        help="Training mode of the model [l2r|r2l|coop]")
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")

    return parser.parse_args(args)


def default_parameters():
    params = tf.contrib.training.HParams(
        input=["", ""],
        output="",
        record="",
        model="transformer",
        # Model sub-type
        is_BPE=False,
        vocab=["", ""],
        # Default training hyper parameters
        num_threads=6,
        batch_size=4096,
        max_length=256,
        length_multiplier=1,
        mantissa_bits=2,
        warmup_steps=4000,
        train_steps=100000,
        buffer_size=10000,
        constant_batch_size=False,
        device_list=[0],
        update_cycle=1,
        initializer="uniform_unit_scaling",
        initializer_gain=1.0,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        clip_grad_norm=5.0,
        learning_rate=1.0,
        learning_rate_decay="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        keep_checkpoint_max=20,
        keep_top_checkpoint_max=5,
        # Validation
        eval_steps=2000,
        eval_secs=0,
        eval_batch_size=32,
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=50,
        validation="",
        references=[""],
        save_checkpoint_secs=0,
        save_checkpoint_steps=1000,
        # Setting this to True can save disk spaces, but cannot restore
        # training using the saved checkpoint
        only_save_trainable=False,
        # Resource config
        allow_growth=True
    )

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(p_name) or not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(p_name) as fd:
        tf.logging.info("Restoring hyper parameters from %s" % p_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def export_params(output_dir, name, params):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)
    with tf.gfile.Open(filename, "w") as fd:
        fd.write(params.to_json())


def collect_params(all_params, params):
    collected = tf.contrib.training.HParams()

    for k in params.values().iterkeys():
        collected.add_hparam(k, getattr(all_params, k))

    return collected


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().iteritems():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().iteritems():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def override_parameters(params, args):
    params.train_mode = args.train_mode
    params.model = args.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.record = args.record or params.record
    params.vocab = args.vocabulary or params.vocab
    params.validation = args.validation or params.validation
    params.references = args.references or params.references
    params.parse(args.parameters)

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(params.vocab[0]),
        "target": vocabulary.load_vocabulary(params.vocab[1])
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    params.vocabulary["target"] = vocabulary.process_vocabulary(
        params.vocabulary["target"], params
    )

    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            control_symbols
        ),
        "target": vocabulary.get_control_mapping(
            params.vocabulary["target"],
            control_symbols
        )
    }

    return params


def get_initializer(params):
    if params.initializer == "uniform":
        max_val = params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="normal")
    elif params.initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)


def get_learning_rate_decay(learning_rate, global_step, params):
    if params.learning_rate_decay in ["linear_warmup_rsqrt_decay", "noam"]:
        step = tf.to_float(global_step)
        warmup_steps = tf.to_float(params.warmup_steps)
        multiplier = params.hidden_size ** -0.5
        decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)

        return learning_rate * decay
    elif params.learning_rate_decay == "piecewise_constant":
        return tf.train.piecewise_constant(tf.to_int32(global_step),
                                           params.learning_rate_boundaries,
                                           params.learning_rate_values)
    elif params.learning_rate_decay == "none":
        return learning_rate
    else:
        raise ValueError("Unknown learning_rate_decay")


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=True)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    config.gpu_options.allow_growth = params.allow_growth
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def decode_target_ids(inputs, params):
    decoded = []
    vocab = params.vocabulary["target"]

    for item in inputs:
        syms = []
        for idx in item:
            if isinstance(idx, six.integer_types):
                sym = vocab[idx]
            else:
                sym = idx

            if sym == params.eos:
                break

            if sym == params.pad:
                break

            syms.append(sym)
        decoded.append(syms)

    return decoded


def get_restore_and_train_var_list(params):
    train_var_mask = []
    restore_var_mask = []
    if params.left2right and params.right2left:
        train_var_mask = []
        restore_var_mask = []
        raise ValueError("Cooperative decoder is not implemented!")
    elif params.left2right:
        train_var_mask = []
        restore_var_mask = []
    elif params.right2left:
        train_var_mask = ["encoder"]
        restore_var_mask = ["right2left_decoder"]

    trainable_vars = tf.trainable_variables()
    trainable_vars_masked = []
    for var in trainable_vars:
        for mask in train_var_mask:
            if mask in var.name:
                trainable_vars_masked.append(var)
                break
    trainable_vars_unmasked = []
    for var in trainable_vars:
        if var not in trainable_vars_masked:
            trainable_vars_unmasked.append(var)

    restore_vars = trainable_vars
    restore_vars_masked = []
    for var in restore_vars:
        for mask in restore_var_mask:
            if mask in var.name:
                restore_vars_masked.append(var)
                break
    restore_vars_unmasked = []
    for var in restore_vars:
        if var not in restore_vars_masked:
            restore_vars_unmasked.append(var)

    return trainable_vars_unmasked, restore_vars_unmasked


def restore_all_variables(checkpoint_dir):
    # Load checkpoints
    tf.logging.info("Loading %s" % checkpoint_dir)
    var_list = tf.train.list_variables(checkpoint_dir)
    reader = tf.train.load_checkpoint(checkpoint_dir)
    values = {}

    for (name, shape) in var_list:
        tensor = reader.get_tensor(name)
        name = name.split(":")[0]
        values[name] = tensor

    var_list = tf.trainable_variables()
    ops = []

    for var in var_list:
        name = var.name.split(":")[0]

        if name in values:
            tf.logging.info("Restore %s" % var.name)
            ops.append(tf.assign(var, values[name]))
    return tf.group(*ops, name="restore_op")


def restore_unmasked_variables(checkpoit_dir, unmask_list):
    tf.logging.info("Loading %s" % checkpoit_dir)
    old_var_list = tf.train.list_variables(checkpoit_dir)
    reader = tf.train.load_checkpoint(checkpoit_dir)
    values = {}

    old_var_list_unmasked = []
    for var in old_var_list:
        for key in unmask_list:
            if key in var[0]:
                old_var_list_unmasked.append(var[0])
                break

    for var_name in old_var_list_unmasked:
        tensor = reader.get_tensor(var_name)
        values[var_name] = tensor

    var_list = tf.trainable_variables()
    ops = []
    for var in var_list:
        name = var.name.split(":")[0]

        if name in old_var_list_unmasked:
            tf.logging.info("Restore %s" % var.name)
            ops.append(tf.assign(var, values[name]))

    return ops


def get_unmasked_vars(unmask_list, trainable_vars):
    trainable_vars_unmasked = []
    for var in trainable_vars:
        for unmask in unmask_list:
            if unmask in var.name:
                trainable_vars_unmasked.append(var)
                break
    return trainable_vars_unmasked


def get_checkpoint_dir(args):
    base_dir = args.output
    if not tf.gfile.Exists(base_dir):
        tf.gfile.MkDir(base_dir)
    if args.train_mode == "coop":
        sub_dir = "coop"
    elif args.train_mode == "l2r":
        sub_dir = "left2right"
    elif args.train_mode == "r2l_only":
        sub_dir = "right2left_only"
    elif args.train_mode == "r2l":
        sub_dir = "right2left"
    else:
        raise ValueError("Unknown training mode %s!" % args.train_mode)
    return os.path.join(base_dir, sub_dir)


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls = models.get_model(args.model)
    params = default_parameters()

    # Import and override parameters
    # Priorities (low -> high):
    # train default -> model default -> saved -> command
    params = merge_parameters(params, model_cls.get_parameters())
    # get checkpoint dir
    checkpoint_dir = get_checkpoint_dir(args)  # e.g. train/left2right
    # params = import_params(checkpoint_dir, args.model, params)
    params = override_parameters(params, args)

    # Print some important parameters for check!!!
    print("\n=== Parameter check ===")
    print("train_mode         : %s" % params.train_mode)
    print("checkpoint_dir     : %s" % checkpoint_dir)
    print("params.left2right  : %s" % params.left2right)
    print("params.right2left  : %s" % params.right2left)
    print("params.batch_size  : %d" % params.batch_size)
    print("params.is_BPE      : %s" % params.is_BPE)
    print("params.update_cycle: %d" % params.update_cycle)
    print("\nPress ANY KEY to continue...")
    sys.stdin.readline()
    # Export all parameters and model specific parameters
    export_params(checkpoint_dir, "params.json", params)
    export_params(
        checkpoint_dir,
        "%s.json" % args.model,
        collect_params(params, model_cls.get_parameters())
    )

    # Build Graph
    with tf.Graph().as_default():
        if not params.record:
            # Build input queue
            features = dataset.get_training_input(params.input, params)
        else:
            features = record.get_input_features(
                os.path.join(params.record, "*train*"), "train", params
            )

        # Cache features for multiple batch update (versus one batch per update)
        update_cycle = params.update_cycle
        features, init_op = cache.cache_features(features,
                                                 update_cycle)

        # Build model
        initializer = get_initializer(params)
        model = model_cls(params)

        # Multi-GPU setting: build multiple GPU loss
        sharded_losses = parallel.parallel_model(
            model.get_training_func(initializer),
            features,
            params.device_list
        )
        loss = tf.add_n(sharded_losses) / len(sharded_losses)

        # Create global step
        global_step = tf.train.get_or_create_global_step()

        # Create learning rate decay strategy
        learning_rate = get_learning_rate_decay(params.learning_rate,
                                                global_step, params)
        learning_rate = tf.convert_to_tensor(learning_rate, dtype=tf.float32)
        tf.summary.scalar("learning_rate", learning_rate)

        # Create optimizer
        if params.optimizer == "Adam":
            opt = tf.train.AdamOptimizer(learning_rate,
                                         beta1=params.adam_beta1,
                                         beta2=params.adam_beta2,
                                         epsilon=params.adam_epsilon)
        elif params.optimizer == "LazyAdam":
            opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate,
                                                   beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2,
                                                   epsilon=params.adam_epsilon)
        else:
            raise RuntimeError("Optimizer %s not supported" % params.optimizer)

        # Get restore_op and trainable_vars_unmasked through train mode

        # 1. Train left2right model
        if params.train_mode == 'l2r':
            # coherence check
            if params.right2left:
                raise ValueError("When in l2r TRAIN MODE, params.right2left should be FALSE!")
            # if there are checkpoint(s) saved, restore from checkpoint
            if tf.gfile.Exists(os.path.join(checkpoint_dir, "checkpoint")):
                restore_op = restore_all_variables(checkpoint_dir)
            # else restore according to model type
            else:
                restore_op = tf.no_op("restore_op")
            trainable_vars_unmasked = tf.trainable_variables()
        # 2. Train right2left model with encoder weights fixed
        elif params.train_mode == 'r2l':
            if params.left2right:
                raise ValueError("When in r2l TRAIN MODE, params.left2right should be FALSE!")
            if tf.gfile.Exists(os.path.join(checkpoint_dir, "checkpoint")):
                restore_op = restore_all_variables(checkpoint_dir)
            else:
                checkpoint_dir_prefix = '/'.join(checkpoint_dir.split('/')[:-1])
                left2right_checkpoint_dir = os.path.join(checkpoint_dir_prefix, 'left2right/eval')
                restore_encoder_ops = restore_unmasked_variables(
                    left2right_checkpoint_dir,
                    ["encoder"]
                )
                restore_op = tf.group(*restore_encoder_ops, name="restore_op")
            trainable_vars_unmasked = get_unmasked_vars(
                ["right2left_decoder"],
                tf.trainable_variables()
            )
        # 2.+ Train right2left model with random initialization
        elif params.train_mode == 'r2l_only':
            if params.left2right:
                raise ValueError("When in r2l_only TRAIN MODE, params.left2right should be FALSE!")
            if tf.gfile.Exists(os.path.join(checkpoint_dir, "checkpoint")):
                restore_op = restore_all_variables(checkpoint_dir)
            else:
                restore_op = tf.no_op("restore_op")
            trainable_vars_unmasked = tf.trainable_variables()
        # 3. Cooperatively train l2r and r2l decoder
        else:
            if params.train_mode != 'coop':
                raise ValueError("Unknown TRAIN MODE %s!" % params.train_mode)
            if not params.left2right or not params.right2left:
                raise ValueError("When in coop TRAIN MODE, both params.left2right and "
                                 "params.right2left should be TRUE!")
            if tf.gfile.Exists(os.path.join(checkpoint_dir, "checkpoint")):
                restore_op = restore_all_variables(checkpoint_dir)
            else:
                checkpoint_dir_prefix = '/'.join(checkpoint_dir.split('/')[:-1])
                l2r_checkpoint_dir = os.path.join(checkpoint_dir_prefix, "left2right/eval")
                r2l_checkpoint_dir = os.path.join(checkpoint_dir_prefix, "right2left/eval")
                restore_encoder_ops = restore_unmasked_variables(
                    r2l_checkpoint_dir,
                    ["encoder"]
                )
                restore_l2r_decoder_ops = restore_unmasked_variables(
                    l2r_checkpoint_dir,
                    ["left2right_decoder"]
                )
                restore_r2l_decoder_ops = restore_unmasked_variables(
                    r2l_checkpoint_dir,
                    ["right2left_decoder"]
                )
                restore_ops = restore_encoder_ops + restore_l2r_decoder_ops + restore_r2l_decoder_ops
                restore_op = tf.group(*restore_ops, name="restore_op")
            trainable_vars_unmasked = tf.trainable_variables()

        # Print parameters
        all_weights = {v.name: v for v in tf.trainable_variables()}
        total_size = 0
        unmasked_size = 0
        print("\nPrint all parameters and their shape!")
        for v_name in sorted(list(all_weights)):
            v = all_weights[v_name]
            tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                            str(v.shape).ljust(20))
            v_size = np.prod(np.array(v.shape.as_list())).tolist()
            total_size += v_size
            if v in trainable_vars_unmasked:
                unmasked_size += v_size
        print("\nPrint unmasked parameters and their shape!")
        unmasked_weights = {v.name: v for v in trainable_vars_unmasked}
        for v_name in sorted(list(unmasked_weights)):
            v = unmasked_weights[v_name]
            tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                            str(v.shape).ljust(20))
        tf.logging.info("Total trainable variables size: %d", total_size)
        tf.logging.info("Unmasked trainable variables size: %d", unmasked_size)
        print("\nConfirm training with mode ===> %s" % params.train_mode)
        print("Press ANY KEY to continue...")
        sys.stdin.readline()

        loss, ops = optimize.create_train_op(
            loss,
            opt,
            trainable_vars_unmasked,
            global_step,
            params
        )

        # Validation
        if params.validation and params.references[0]:
            files = [params.validation] + list(params.references)
            eval_inputs = dataset.sort_and_zip_files(files)
            eval_input_fn = dataset.get_evaluation_input
        else:
            eval_input_fn = None

        # Add hooks
        save_vars = tf.trainable_variables() + [global_step]
        saver = tf.train.Saver(
            var_list=save_vars if params.only_save_trainable else None,
            max_to_keep=params.keep_checkpoint_max,
            sharded=False
        )
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        multiplier = tf.convert_to_tensor([update_cycle, 1])

        train_hooks = [
            tf.train.StopAtStepHook(last_step=params.train_steps),
            tf.train.NanTensorHook(loss),
            tf.train.LoggingTensorHook(
                {
                    "step": global_step,
                    "loss": loss,
                    "source": tf.shape(features["source"]) * multiplier,
                    "target": tf.shape(features["target"]) * multiplier
                },
                every_n_iter=50
            ),
            tf.train.CheckpointSaverHook(
                checkpoint_dir=checkpoint_dir,
                save_secs=params.save_checkpoint_secs or None,
                save_steps=params.save_checkpoint_steps or None,
                saver=saver
            )
        ]

        config = session_config(params)

        if eval_input_fn is not None:
            train_hooks.append(
                hooks.EvaluationHook(
                    lambda f: inference.create_inference_graph(
                        [model.get_inference_func()], f, params
                    ),
                    lambda: eval_input_fn(eval_inputs, params),
                    lambda x: decode_target_ids(x, params),
                    checkpoint_dir,
                    params.train_mode == 'r2l' or params.train_mode == 'r2l_only',  # is_Reverse
                    params.is_BPE,
                    config,  # for create a chief session
                    params.keep_top_checkpoint_max,
                    eval_secs=params.eval_secs,
                    eval_steps=params.eval_steps
                )
            )

        # Create session, do not use default CheckpointSaverHook
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=checkpoint_dir, hooks=train_hooks,
                save_checkpoint_secs=None, config=config) as sess:
            # Restore pre-trained variables
            sess._tf_sess().run(restore_op)  # if use sess.run(...) global_step would not increase 1

            while not sess.should_stop():
                sess._tf_sess().run([init_op, ops["zero_op"]])
                for i in range(params.update_cycle - 1):
                    sess._tf_sess().run(ops["collect_op"])
                sess.run(ops["train_op"])


if __name__ == "__main__":
    main(parse_args())
