from diffuser.utils import watch

# ------------------------ base ------------------------ #

args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("n_diffusion_steps", "K"),
    ("diffusion", "D"),
]

logbase = "logs"

base = {
    "diffusion": {
        # model
        "model": "models.UNet1DTemporalCondModel",
        "diffusion": "models.GaussianDiffusion",
        "horizon": 8,
        "n_diffusion_steps": 20,
        "loss_type": "l2",
        "loss_discount": 1.0,
        "returns_condition": False,
        "action_weight": 10,
        "dim": 32,
        "dim_mults": (1, 2, 4, 8),
        "predict_epsilon": True,
        "dynamic_loss": False,
        "hidden_dim": 256,
        "attention": False,
        "condition_dropout": 0.25,
        "condition_guidance_w": 1.2,
        "test_ret": 0.9,
        # dataset
        "loader": "datasets.SequenceDataset",
        "normalizer": "LimitsNormalizer",
        "preprocess_fns": [],
        "clip_denoised": False,
        "use_padding": True,
        "max_path_length": 150,
        "include_returns": True,
        "returns_scale": 400,
        "discount": 0.99,
        # serialization
        "logbase": logbase,
        "prefix": "diffusion/",
        "exp_name": watch(args_to_watch),
        # training
        "n_steps_per_epoch": 1000,
        "n_train_steps": 1e5,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.995,
        "train_test_split": 0.9,
        "device": "cuda",
        "seed": 0,
    },
    "plan": {
        "policy": "sampling.Policy",
        "max_episode_length": 200,
        "batch_size": 4,
        "preprocess_fns": [],
        "device": "cuda",
        "seed": 0,
        "test_ret": 0,
        # serialization
        "loadbase": None,
        "logbase": logbase,
        "prefix": "plans/",
        "exp_name": watch(args_to_watch),
        # diffusion model
        "diffusion": "models.GaussianDiffusion",
        "horizon": 8,
        "n_diffusion_steps": 20,
        "returns_condition": False,
        "predict_epsilon": True,
        "dynamic_loss": False,
        # loading
        "diffusion_loadpath": "f:diffusion/H{horizon}_K{n_diffusion_steps}_D{diffusion}",
        "value_loadpath": "f:values/H{horizon}_K{n_diffusion_steps}",
        "diffusion_epoch": "best",
        "verbose": False,
        "suffix": "0",
    },
}
