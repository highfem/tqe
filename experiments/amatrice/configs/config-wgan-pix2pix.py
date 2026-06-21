import ml_collections


def new_dict(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  config = ml_collections.ConfigDict()
  config.rng_key = 123
  config.model_type = "latent_wgan"

  batch_size_per_gpu = 128
  n_out_channels = 3
  n_latent_channels = 8
  stft_channels = 256
  max_len = 8161
  hop_size = 32

  config.model = new_dict(
    gan=new_dict(lamb=5),
    nn=new_dict(
      name="pix2pix",
      generator=new_dict(
        nn=new_dict(
          n_out_channels=n_latent_channels,
          n_channels=128,
          n_embedding_dimension=256,
          append_noise_in_generator=False,
          dropout=0.5,
        ),
      ),
      critic=new_dict(
        nn=new_dict(
          n_out_channels=n_latent_channels,
          n_channels=128,
          n_embedding_dimension=256,
          dropout=0.1,
          use_patch_gan=True,
        ),
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
    n_steps=300_000,
    batch_size=batch_size_per_gpu * 4,
    buffer_size=batch_size_per_gpu * 4 * 2,
    prefetch_size=4,
    do_reshuffle=True,
    percentage_data_as_validation_set=0.1,
    early_stopping=new_dict(n_patience=100, min_delta=0.001),
    checkpoints=new_dict(
      max_to_keep=10,
      save_interval_steps=1,
    ),
    n_eval_frequency=5_000,
    n_eval_batches=25,
    n_checkpointing_frequency=20_000,
    n_sampling_frequency=10_000,
    n_sampling_batches=5,
    ema_rate=0.999,
    n_update_generator=5,
  )

  config.optimizer = new_dict(
    name="adamw",
    params=new_dict(
      learning_rate=0.002,
      do_warmup=True,
      warmup_steps=10_000,
      do_decay=True,
      decay_steps=300_000,
      end_learning_rate=1e-6,
      do_gradient_clipping=True,
      gradient_clipping=1.0,
      b1=0.5,
      b2=0.9,
      weight_decay=1e-5,
    ),
  )

  return config
