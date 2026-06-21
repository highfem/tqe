import ml_collections


def new_dict(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  config = ml_collections.ConfigDict()
  config.rng_key = 123
  config.model_type = "latent_consistency_distillation"

  batch_size_per_gpu = 64
  n_out_channels = 3
  n_latent_channels = 8
  stft_channels = 256
  max_len = 8161
  hop_size = 32

  config.model = new_dict(
    name="edm",
    distillation=new_dict(
      type="edm_style",
    ),
    sampler=new_dict(
      n_steps=2,
    ),
    nn=new_dict(
      name="dit",
      dit_score_net=new_dict(
        n_out_channels=n_latent_channels,
        n_channels=768,  # 768, 1024
        patch_size=2,
        n_blocks=12,  # 12, 16
        n_heads=12,  # 12, 16
        dropout_rate=0.0,
        n_embedding_dimension=256,
        do_context_projections=True,
        do_context_conditioning=True,
        do_meta_conditioning=True,
      ),
    ),
  )

  config.data = new_dict(
    dataset="amatrice_paired_waveforms",
    n_out_channels=n_out_channels,
    image_size=(stft_channels // 2, max_len // hop_size + 1),
    condition_length=8,
  )

  config.representation = new_dict(
    representations=[
      new_dict(
        name="log_spectrogram",
        params=new_dict(
          stft_channels=stft_channels,
          hop_size=hop_size,
          max_len=max_len,
        ),
      ),
    ],
    latent_channels=n_latent_channels,
    # 24005/9000 is the sequence length
    # settig 23521/8161 makes es it that the dimension after STFT is 736/256
    # this allows doing 3/4/5 average pools and results in even dimensionalities
    # so that upsampling exactly retains the dimension. since we use TF for the
    max_len=max_len,
    fs=100,
  )

  config.callbacks = [
    new_dict(
      name="amplitude_spectral_density", params=new_dict(channel=i, fs=100)
    )
    for i in range(3)
  ]

  config.training = new_dict(
    n_steps=100_000,
    batch_size=batch_size_per_gpu * 4,
    buffer_size=batch_size_per_gpu * 4 * 2,
    prefetch_size=4,
    do_reshuffle=True,
    percentage_data_as_validation_set=0.05,
    early_stopping=new_dict(n_patience=100, min_delta=0.001),
    checkpoints=new_dict(
      max_to_keep=10,
      save_interval_steps=1,
    ),
    n_eval_frequency=5_000,
    n_eval_batches=25,
    n_checkpointing_frequency=10_000,
    n_sampling_frequency=10_000,
    n_sampling_batches=5,
    ema_rate=0.9999,
    distillation_stages=5000,
  )

  config.optimizer = new_dict(
    name="adamw",
    params=new_dict(
      learning_rate=1e-4,
      do_warmup=True,
      warmup_steps=1000,
      do_decay=True,
      decay_steps=100_000,
      end_learning_rate=1e-6,
      do_gradient_clipping=True,
      gradient_clipping=1.0,
      b1=0.9,
      b2=0.999,
      weight_decay=0.00001,
    ),
  )

  return config
