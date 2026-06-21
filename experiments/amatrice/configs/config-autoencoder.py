import ml_collections


def new_dict(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  config = ml_collections.ConfigDict()
  config.rng_key = 1
  config.model_type = "target_encoder"

  batch_size_per_gpu = 128
  n_channels = 64

  n_out_channels = 3
  n_latent_channels = 8
  stft_channels = 256
  max_len = 8161
  hop_size = 32

  config.model = new_dict(
    nn=new_dict(
      base_model=new_dict(
        n_channels=n_channels,
        channel_multipliers=(1, 2, 2, 4),
        downsampling_strides=((2, 2), (2, 2), (1, 2)),
        n_resnet_blocks=3,
        attention_resolutions=(),
        kernel_size=3,
        dropout_rate=0.1,
        n_groups=32,
      ),
      encoder=new_dict(
        # for shift and scale of Gaussians
        n_out_channels=n_latent_channels * 2,
      ),
      decoder=new_dict(n_out_channels=n_out_channels),
    )
  )

  config.data = new_dict(
    dataset="amatrice_paired_waveforms",
    n_out_channels=n_out_channels,
    image_size=(stft_channels // 2, max_len // hop_size + 1),
  )

  config.callbacks = [
    new_dict(
      name="amplitude_spectral_density", params=new_dict(channel=i, fs=100)
    )
    for i in range(3)
  ]

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

  config.training = new_dict(
    params=new_dict(
      kl_weight=1e-6,
    ),
    n_steps=200_000,
    batch_size=batch_size_per_gpu * 4,
    buffer_size=batch_size_per_gpu * 4 * 2,
    prefetch_size=4,
    do_reshuffle=True,
    percentage_data_as_validation_set=0.1,
    early_stopping=new_dict(n_patience=20, min_delta=0.001),
    checkpoints=new_dict(
      max_to_keep=10,
      save_interval_steps=1,
    ),
    n_eval_frequency=5_000,
    n_eval_batches=10,
    n_checkpointing_frequency=5_000,
    n_sampling_frequency=10_000,
    n_sampling_batches=5,
  )

  config.optimizer = new_dict(
    name="adamw",
    params=new_dict(
      learning_rate=1e-4,
      do_warmup=False,
      warmup_steps=1000,
      do_decay=True,
      decay_steps=300_000,
      end_learning_rate=1e-6,
      do_gradient_clipping=False,
      gradient_clipping=1.0,
      b1=0.9,
      b2=0.999,
      weight_decay=0.00001,
    ),
  )

  return config
