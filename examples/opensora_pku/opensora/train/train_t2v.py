import logging
import math
import os
import sys

import yaml

import mindspore as ms
from mindspore import Model, nn
from mindspore.train.callback import TimeMonitor

mindone_lib_path = os.path.abspath(os.path.abspath("../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.append("./")

from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
from opensora.dataset.t2v_dataset import create_dataloader
from opensora.models.ae import ae_channel_config, ae_stride_config, getae_wrapper
from opensora.models.ae.videobase.modules.updownsample import TrilinearInterpolate
from opensora.models.diffusion.diffusion import create_diffusion_T as create_diffusion
from opensora.models.diffusion.latte.modeling_latte import Latte_models, LayerNorm
from opensora.models.diffusion.latte.modules import Attention
from opensora.models.diffusion.latte.net_with_loss import DiffusionWithLoss
from opensora.models.text_encoder.t5 import T5Embedder
from opensora.train.commons import create_loss_scaler, init_env, parse_args
from opensora.utils.utils import get_precision

from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallbackEpoch
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"
os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"
ms.context.set_context(jit_config={"jit_level": "O1"})  # O0: KBK, O1:DVM, O2: GE
logger = logging.getLogger(__name__)


def set_all_reduce_fusion(
    params,
    split_num: int = 7,
    distributed: bool = False,
    parallel_mode: str = "data",
) -> None:
    """Set allreduce fusion strategy by split_num."""

    if distributed and parallel_mode == "data":
        all_params_num = len(params)
        step = all_params_num // split_num
        split_list = [i * step for i in range(1, split_num)]
        split_list.append(all_params_num - 1)
        logger.info(f"Distribute config set: dall_params_num: {all_params_num}, set all_reduce_fusion: {split_list}")
        ms.set_auto_parallel_context(all_reduce_fusion_config=split_list)


def main(args):
    # 1. init
    save_src_strategy = args.use_parallel and args.parallel_mode != "data"
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        mempool_block_size=args.mempool_block_size,
        global_bf16=args.global_bf16,
        strategy_ckpt_save_file=os.path.join(args.output_dir, "src_strategy.ckpt") if save_src_strategy else "",
        optimizer_weight_shard_size=args.optimizer_weight_shard_size,
        sp_size=args.sp_size,
    )
    set_logger(name="", output_dir=args.output_dir, rank=rank_id, log_level=eval(args.log_level))

    train_with_vae_latent = args.vae_latent_folder is not None and len(args.vae_latent_folder) > 0
    if train_with_vae_latent:
        assert os.path.exists(
            args.vae_latent_folder
        ), f"The provided vae latent folder {args.vae_latent_folder} is not existent!"
        logger.info("Train with vae latent cache.")
        vae = None
    else:
        logger.info("vae init")
        vae = getae_wrapper(args.ae)(args.ae_path, subfolder="vae")
        vae_dtype = get_precision(args.vae_precision)
        if vae_dtype == ms.float16:
            custom_fp32_cells = [nn.GroupNorm] if args.vae_keep_gn_fp32 else []
        else:
            custom_fp32_cells = [nn.AvgPool2d, TrilinearInterpolate]
        vae = auto_mixed_precision(vae, amp_level="O2", dtype=vae_dtype, custom_fp32_cells=custom_fp32_cells)
        logger.info(f"Use amp level O2 for causal 3D VAE with dtype={vae_dtype}, custom_fp32_cells {custom_fp32_cells}")

        vae.set_train(False)
        for param in vae.get_parameters():  # freeze vae
            param.requires_grad = False
        if args.enable_tiling:
            vae.vae.enable_tiling()
            vae.vae.tile_overlap_factor = args.tile_overlap_factor

        ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
        args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = ae_stride_t, ae_stride_h, ae_stride_w
        args.ae_stride = args.ae_stride_h
        patch_size = args.model[-3:]
        patch_size_t, patch_size_h, patch_size_w = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
        args.patch_size = patch_size_h
        args.patch_size_t, args.patch_size_h, args.patch_size_w = patch_size_t, patch_size_h, patch_size_w
        assert (
            ae_stride_h == ae_stride_w
        ), f"Support only ae_stride_h == ae_stride_w now, but found ae_stride_h ({ae_stride_h}), ae_stride_w ({ae_stride_w})"
        assert (
            patch_size_h == patch_size_w
        ), f"Support only patch_size_h == patch_size_w now, but found patch_size_h ({patch_size_h}), patch_size_w ({patch_size_w})"
        assert (
            args.max_image_size % ae_stride_h == 0
        ), f"Image size must be divisible by ae_stride_h, but found max_image_size ({args.max_image_size}), "
        " ae_stride_h ({ae_stride_h})."

        latent_size = (args.max_image_size // ae_stride_h, args.max_image_size // ae_stride_w)
        vae.latent_size = latent_size
        args.stride_t = ae_stride_t * patch_size_t
        args.stride = ae_stride_h * patch_size_h

    logger.info(f"Init Latte T2V model: {args.model}")
    ae_time_stride = 4
    video_length = args.num_frames // ae_time_stride + 1
    FA_dtype = get_precision(args.precision) if get_precision(args.precision) != ms.float32 else ms.bfloat16
    assert not args.multi_scale, "Multi-scale training is not supported now!"
    assert (
        args.compress_kv_factor >= 1
    ), f"Expect that compress_kv_factor is greater than zero, but got {args.compress_kv_factor}"
    assert (
        args.num_no_recompute >= 0 and args.num_no_recompute <= 28
    ), f"Expect that the number of no recomputation is within [0, the total number of transformer blocks (28)], but got {args.num_no_recompute}"
    latte_model = Latte_models[args.model](
        in_channels=ae_channel_config[args.ae],
        out_channels=ae_channel_config[args.ae] * 2,
        attention_bias=True,
        sample_size=latent_size,
        num_vector_embeds=None,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        use_linear_projection=False,
        only_cross_attention=False,
        double_self_attention=False,
        upcast_attention=False,
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        attention_type="default",
        video_length=video_length,
        enable_flash_attention=args.enable_flash_attention,
        use_recompute=args.use_recompute,
        compress_kv_factor=args.compress_kv_factor,
        use_rope=args.use_rope,
        model_max_length=args.model_max_length,
        FA_dtype=FA_dtype,
        num_no_recompute=args.num_no_recompute,
    )

    # mixed precision
    if args.precision == "fp32":
        model_dtype = get_precision(args.precision)
    else:
        model_dtype = get_precision(args.precision)
        if not args.global_bf16:
            if model_dtype == ms.float16:
                custom_fp32_cells = [LayerNorm, Attention, nn.SiLU, nn.GELU]
            else:
                custom_fp32_cells = [nn.MaxPool2d, LayerNorm, nn.SiLU, nn.GELU]
            latte_model = auto_mixed_precision(
                latte_model,
                amp_level=args.amp_level,
                dtype=model_dtype,
                custom_fp32_cells=custom_fp32_cells,
            )
            logger.info(
                f"Set mixed precision to {args.amp_level} with dtype={args.precision}, custom_fp32_cells: {custom_fp32_cells}"
            )
        else:
            logger.info(f"Using global bf16 for latte t2v model. Force model dtype from {model_dtype} to ms.bfloat16")
            model_dtype = ms.bfloat16
    # load checkpoint
    if args.pretrained is not None and len(args.pretrained) > 0:
        assert os.path.exists(args.pretrained), f"Provided checkpoint file {args.pretrained} does not exist!"
        logger.info(f"Loading ckpt {args.pretrained}...")
        latte_model.load_from_checkpoint(args.pretrained)
    else:
        logger.info("Use random initialization for Latte")
    latte_model.set_train(True)

    if not args.text_embed_cache:
        logger.info("T5 init")
        text_encoder = T5Embedder(
            dir_or_name=args.text_encoder_name,
            cache_dir="./",
            model_max_length=args.model_max_length,
        )
        # mixed precision
        text_encoder_dtype = get_precision(args.text_encoder_precision)  # using bf16 for text encoder and vae
        text_encoder = auto_mixed_precision(text_encoder, amp_level="O2", dtype=text_encoder_dtype)
        text_encoder.dtype = text_encoder_dtype
        logger.info(f"Use amp level O2 for text encoder T5 with dtype={text_encoder_dtype}")

        tokenizer = text_encoder.tokenizer
    else:
        text_encoder = None
        tokenizer = None
        text_encoder_dtype = None

    # 2.3 ldm with loss
    diffusion = create_diffusion(timestep_respacing="")
    assert args.use_image_num >= 0, f"Expect to have use_image_num>=0, but got {args.use_image_num}"
    if args.use_image_num > 0:
        logger.info("Enable video-image-joint training")
        if args.use_img_from_vid:
            args.image_data = ""
    else:
        logger.info("Training on video datasets only.")
        args.image_data = ""
    latent_diffusion_with_loss = DiffusionWithLoss(
        latte_model,
        diffusion,
        vae=vae,
        text_encoder=text_encoder,
        text_emb_cached=args.text_embed_cache,
        video_emb_cached=False,
        use_image_num=args.use_image_num,
        dtype=model_dtype,
    )
    split_time_upsample = True
    assert not (
        args.num_frames % 2 == 0 and split_time_upsample
    ), "num of frames must be odd if split_time_upsample is True"
    # 3. create dataset
    assert args.dataset == "t2v", "Support t2v dataset only."
    ds_config = dict(
        image_data=args.image_data,
        video_data=args.video_data,
        sample_size=args.max_image_size,
        num_frames=args.num_frames,
        tokenizer=tokenizer,
        return_text_emb=args.text_embed_cache,
        disable_flip=not args.enable_flip,
        use_image_num=args.use_image_num,
        use_img_from_vid=args.use_img_from_vid,
        model_max_length=args.model_max_length,
        filter_nonexistent=args.filter_nonexistent,
    )
    dataset = create_dataloader(
        ds_config,
        batch_size=args.batch_size,
        shuffle=True,
        device_num=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
        rank_id=rank_id if not get_sequence_parallel_state() else hccl_info.group_id,
        num_parallel_workers=args.dataloader_num_workers,
        max_rowsize=args.max_rowsize,
        prefetch_size=args.dataloader_prefetch_size,
    )
    dataset_size = dataset.get_dataset_size()
    assert dataset_size > 0, "Incorrect dataset size. Please check your dataset size and your global batch size"

    # 4. build training utils: lr, optim, callbacks, trainer
    if args.scale_lr:
        learning_rate = args.start_learning_rate * args.batch_size * args.gradient_accumulation_steps * device_num
        end_learning_rate = args.end_learning_rate * args.batch_size * args.gradient_accumulation_steps * device_num
    else:
        learning_rate = args.start_learning_rate
        end_learning_rate = args.end_learning_rate

    if args.dataset_sink_mode and args.sink_size != -1:
        assert args.sink_size > 0, f"Expect that sink size is a positive integer, but got {args.sink_size}"
        steps_per_sink = args.sink_size
    else:
        steps_per_sink = dataset_size

    if args.max_train_steps is not None:
        assert args.max_train_steps > 0, f"max_train_steps should a positive integer, but got {args.max_train_steps}"
        total_train_steps = args.max_train_steps
        args.epochs = math.ceil(total_train_steps / dataset_size)
    else:
        # use args.epochs
        assert (
            args.epochs is not None and args.epochs > 0
        ), f"When args.max_train_steps is not provided, args.epochs must be a positive integer! but got {args.epochs}"
        total_train_steps = args.epochs * dataset_size

    sink_epochs = math.ceil(total_train_steps / steps_per_sink)
    total_train_steps = sink_epochs * steps_per_sink

    if steps_per_sink == dataset_size:
        logger.info(
            f"Number of training steps: {total_train_steps}; Number of epochs: {args.epochs}; Number of batches in a epoch (dataset_size): {dataset_size}"
        )
    else:
        logger.info(
            f"Number of training steps: {total_train_steps}; Number of sink epochs: {sink_epochs}; Number of batches in a sink (sink_size): {steps_per_sink}"
        )

    if args.checkpointing_steps is None:
        ckpt_save_interval = args.ckpt_save_interval
        step_mode = False
    else:
        step_mode = not args.dataset_sink_mode
        if not args.dataset_sink_mode:
            ckpt_save_interval = args.checkpointing_steps
        else:
            # still need to count interval in sink epochs
            ckpt_save_interval = max(1, args.checkpointing_steps // steps_per_sink)
            if args.checkpointing_steps % steps_per_sink != 0:
                logger.warning(
                    f"`checkpointing_steps` must be times of sink size or dataset_size under dataset sink mode."
                    f"Checkpoint will be saved every {ckpt_save_interval * steps_per_sink} steps."
                )
    if step_mode != args.step_mode:
        logger.logging("Using args.checkpointing_steps to determine whether to use step mode to save ckpt.")
        if args.checkpointing_steps is None:
            logger.warning(f"args.checkpointing_steps is not provided. Force step_mode to {step_mode}!")
        else:
            logger.warning(
                f"args.checkpointing_steps is provided. data sink mode is {args.dataset_sink_mode}. Force step mode to {step_mode}!"
            )
    logger.info(
        "ckpt_save_interval: {} {}".format(
            ckpt_save_interval, "steps" if (not args.dataset_sink_mode and step_mode) else "sink epochs"
        )
    )
    # build learning rate scheduler
    if not args.lr_decay_steps:
        args.lr_decay_steps = total_train_steps - args.lr_warmup_steps  # fix lr scheduling
        if args.lr_decay_steps <= 0:
            logger.warning(
                f"decay_steps is {args.lr_decay_steps}, please check epochs, dataset_size and warmup_steps. "
                f"Will force decay_steps to be set to 1."
            )
            args.lr_decay_steps = 1
    assert (
        args.lr_warmup_steps >= 0
    ), f"Expect args.lr_warmup_steps to be no less than zero,  but got {args.lr_warmup_steps}"

    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        name=args.lr_scheduler,
        lr=learning_rate,
        end_lr=end_learning_rate,
        warmup_steps=args.lr_warmup_steps,
        decay_steps=args.lr_decay_steps,
        total_steps=total_train_steps,
    )
    set_all_reduce_fusion(
        latent_diffusion_with_loss.trainable_params(),
        split_num=7,
        distributed=args.use_parallel,
        parallel_mode=args.parallel_mode,
    )

    # build optimizer
    assert args.optim.lower() == "adamw", f"Not support optimizer {args.optim}!"
    optimizer = create_optimizer(
        latent_diffusion_with_loss.trainable_params(),
        name=args.optim,
        betas=args.betas,
        eps=args.optim_eps,
        group_strategy=args.group_strategy,
        weight_decay=args.weight_decay,
        lr=lr,
    )

    loss_scaler = create_loss_scaler(args)
    # resume ckpt
    ckpt_dir = os.path.join(args.output_dir, "ckpt")
    start_epoch = 0
    if args.resume_from_checkpoint:
        resume_ckpt = (
            os.path.join(ckpt_dir, "train_resume.ckpt")
            if isinstance(args.resume_from_checkpoint, bool)
            else args.resume_from_checkpoint
        )

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
            latte_model, optimizer, resume_ckpt
        )
        loss_scaler.loss_scale_value = loss_scale
        loss_scaler.cur_iter = cur_iter
        loss_scaler.last_overflow_iter = last_overflow_iter

    # trainer (standalone and distributed)
    ema = (
        EMA(
            latent_diffusion_with_loss.network,
            ema_decay=0.9999,
        )
        if args.use_ema
        else None
    )
    assert (
        args.gradient_accumulation_steps > 0
    ), f"Expect gradient_accumulation_steps is a positive integer, but got {args.gradient_accumulation_steps}"
    net_with_grads = TrainOneStepWrapper(
        latent_diffusion_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    if not args.global_bf16:
        model = Model(net_with_grads)
    else:
        model = Model(net_with_grads, amp_level="O0")
    # callbacks
    callback = [TimeMonitor(args.log_interval)]
    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)

    if args.parallel_mode == "optim":
        cb_rank_id = None
        ckpt_save_dir = os.path.join(ckpt_dir, f"rank_{rank_id}")
        output_dir = os.path.join(args.output_dir, "log", f"rank_{rank_id}")
        if args.ckpt_max_keep != 1:
            logger.warning("For semi-auto parallel training, the `ckpt_max_keep` is force to be 1.")
        ckpt_max_keep = 1
        integrated_save = False
        save_training_resume = False  # TODO: support training resume
    else:
        cb_rank_id = rank_id
        ckpt_save_dir = ckpt_dir
        output_dir = None
        ckpt_max_keep = args.ckpt_max_keep
        integrated_save = True
        save_training_resume = True

    if rank_id == 0 or args.parallel_mode == "optim":
        save_cb = EvalSaveCallback(
            network=latent_diffusion_with_loss.network,
            rank_id=cb_rank_id,
            ckpt_save_dir=ckpt_save_dir,
            output_dir=output_dir,
            ema=ema,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=ckpt_max_keep,
            step_mode=step_mode,
            use_step_unit=(args.checkpointing_steps is not None),
            ckpt_save_interval=ckpt_save_interval,
            log_interval=args.log_interval,
            start_epoch=start_epoch,
            model_name=args.model.replace("/", "-"),
            record_lr=False,
            integrated_save=integrated_save,
            save_training_resume=save_training_resume,
        )
        callback.append(save_cb)
        if args.profile:
            callback.append(ProfilerCallbackEpoch(2, 2, "./profile_data"))

    # 5. log and save config
    if rank_id == 0:
        if vae is not None:
            num_params_vae, num_params_vae_trainable = count_params(vae)
        else:
            num_params_vae, num_params_vae_trainable = 0, 0
        num_params_latte, num_params_latte_trainable = count_params(latte_model)
        num_params = num_params_vae + num_params_latte
        num_params_trainable = num_params_vae_trainable + num_params_latte_trainable
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}"
                + (f"\nParallel mode: {args.parallel_mode}" if args.use_parallel else ""),
                f"Num params: {num_params:,} (latte: {num_params_latte:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_params_trainable:,}",
                f"Transformer model dtype: {model_dtype}",
                f"Transformer AMP level: {args.amp_level}" if not args.global_bf16 else "Global BF16: True",
                f"VAE dtype: {vae_dtype} (amp level O2)"
                + (
                    f"\nText encoder dtype: {text_encoder_dtype} (amp level O2)"
                    if text_encoder_dtype is not None
                    else ""
                ),
                f"Learning rate: {learning_rate}",
                f"Batch size: {args.batch_size}",
                f"Image size: {args.max_image_size}",
                f"Number of frames: {args.num_frames}",
                f"Use image num: {args.use_image_num}",
                f"Optimizer: {args.optim}",
                f"Optimizer epsilon: {args.optim_eps}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num of training steps: {total_train_steps}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"EMA decay: {args.ema_decay}",
                f"Enable flash attention: {args.enable_flash_attention} ({FA_dtype})",
                f"Use recompute: {args.use_recompute}",
                f"Dataset sink: {args.dataset_sink_mode}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")

        with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    # 6. train
    model.train(
        sink_epochs,
        dataset,
        callbacks=callback,
        dataset_sink_mode=args.dataset_sink_mode,
        sink_size=args.sink_size,
        initial_epoch=start_epoch,
    )


def parse_t2v_train_args(parser):
    parser.add_argument("--output_dir", default="outputs/", help="The directory where training results are saved.")
    parser.add_argument("--dataset", type=str, default="t2v")
    parser.add_argument("--image_data", type=str, required=True)
    parser.add_argument("--video_data", type=str, required=True)
    parser.add_argument(
        "--filter_nonexistent",
        type=str2bool,
        default=True,
        help="Whether to filter out non-existent samples in image datasets and video datasets." "Defaults to True.",
    )
    parser.add_argument(
        "--text_embed_cache",
        type=str2bool,
        default=True,
        help="Whether to use T5 embedding cache. Must be provided in image/video_data.",
    )
    parser.add_argument("--vae_latent_folder", default=None, type=str, help="root dir for the vae latent data")
    parser.add_argument("--model", type=str, default="LatteT2V-XL/122")
    parser.add_argument("--ae", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--ae_path", type=str, default="LanguageBind/Open-Sora-Plan-v1.1.0")

    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--max_image_size", type=int, default=512)
    parser.add_argument("--compress_kv", action="store_true")
    parser.add_argument("--compress_kv_factor", type=int, default=1)
    parser.add_argument("--use_rope", action="store_true")
    parser.add_argument("--pretrained", type=str, default=None)

    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--enable_tiling", action="store_true")

    parser.add_argument("--text_encoder_name", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--model_max_length", type=int, default=300)
    parser.add_argument("--multi_scale", action="store_true")

    # parser.add_argument("--enable_tracker", action="store_true")
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )

    parser.add_argument(
        "--enable_flip",
        action="store_true",
        help="enable random flip video (disable it to avoid motion direction and text mismatch)",
    )
    parser.add_argument(
        "--num_no_recompute",
        type=int,
        default=0,
        help="If use_recompute is True, `num_no_recompute` blocks will be removed from the recomputation list."
        "This is a positive integer which can be tuned based on the memory usage.",
    )
    parser.add_argument("--dataloader_prefetch_size", type=int, default=None, help="minddata prefetch size setting")
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--vae_keep_gn_fp32",
        default=False,
        type=str2bool,
        help="whether keep GroupNorm in fp32. Defaults to False in inference mode. If training vae, better set it to True",
    )
    parser.add_argument(
        "--vae_precision",
        default="fp16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for vae. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--text_encoder_precision",
        default="bf16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for T5 text encoder. Default is `bf16`, which corresponds to ms.bfloat16",
    )
    parser.add_argument(
        "--enable_parallel_fusion", default=True, type=str2bool, help="Whether to parallel fusion for AdamW"
    )
    return parser


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args(additional_parse_args=parse_t2v_train_args)
    main(args)
