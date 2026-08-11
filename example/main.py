import os 
from datetime import datetime
from pathlib import Path
#from tellusfm.network_light import Senseiver
import torch 
import torch.distributed as dist
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

import tellusfm as tfm 

if __name__ == '__main__':

    # --------------------------------------------------
    # 1. Parse command-line arguments
    # --------------------------------------------------
    args = tfm.parse_command_line() 

    # --------------------------------------------------
    # 2. Initialize logging system
    #    - Sets up log file
    #    - Ensures all future prints go through logging
    # --------------------------------------------------
    tfm.initialize_log_file(args) 

    # --------------------------------------------------
    # 3. Read experiment configuration from YAML
    #    - Merge with defaults if needed
    #    - Handles verbose flag for pretty-printing
    # --------------------------------------------------
    # Step 1: Read and preprocess config
    tfm.local_print_log("Reading configuration file...")
    config = tfm.read_yaml_config(args.config)
    validation_summary = tfm.validate_config(config)
    for warning in validation_summary["warnings"]:
        tfm.local_print_log(f"Config warning: {warning}")

    # --------------------------------------------------
    # 4. Extract configuration sections
    #    - Data parameters
    #    - Encoder hyperparameters
    #    - Decoder hyperparameters
    # --------------------------------------------------
    # Step 2: Build sub-configurations
    tfm.local_print_log("Loading model parameters...")
    data_config = tfm.create_section_config(config, "datasets", verbose=args.verbose)
    encoder_config = tfm.create_section_config(config, "encoder", verbose=args.verbose)
    decoder_config = tfm.create_section_config(config, "decoder", verbose=args.verbose)
    model_config = tfm.create_section_config(config, "model_params", verbose=args.verbose)
    embeddings_config = tfm.create_section_config(config, "embeddings", verbose=args.verbose)
    checkpoints_config = tfm.create_section_config(config, "checkpoints", verbose=args.verbose)
    rule_based_config = tfm.create_section_config(config, "rule_based_params", verbose=args.verbose)
    run_type = str(checkpoints_config.get("run_type", "train")).lower()
    model_config["run_type"] = run_type
    if args.test_bc is not None:
        model_config["test_bc"] = args.test_bc

    tfm.local_print_log("Loading model parameters complete!")

    # Step 3: Debug print or pass to downstream model initialization
    if args.verbose:
        print("\n--- Data Config ---")
        print(data_config)
        print("\n--- Encoder Config ---")
        print(encoder_config)
        print("\n--- Decoder Config ---")
        print(decoder_config)
        print("\n--- Model Config ---")
        print(model_config)
        print("\n--- Checkpoint Config ---")
        print(checkpoints_config)

    # --------------------------------------------------
    # 5. Runtime setup
    #    - Set random seed for reproducibility
    #    - Clear CUDA cache before training
    # --------------------------------------------------
    seed_everything(model_config['seed'], workers=True) 
    torch.cuda.empty_cache() 

    # load the simulation data and create a dataloader
    #set_num_workers = os.cpu_count()
    print("Number of workers: ", model_config['num_workers'])
    print("Simulation type: ", model_config["sim_type"]) 

    dataloader = None
    val_dataloader = None
    test_dataloader = None
    if run_type == "test":
        test_dataloader = tfm.create_senseiver_dataloader(train=False, model_config = model_config,
                                                 embeddings_config = embeddings_config,
                                                 data_config = data_config, rule_based_params=rule_based_config,
                                                 run_type = "test")
    else:
        dataloader = tfm.create_senseiver_dataloader(train=True, model_config = model_config,
                                                 embeddings_config = embeddings_config,
                                                 data_config = data_config, rule_based_params=rule_based_config,
                                                 run_type = run_type)

        val_dataloader = tfm.create_senseiver_dataloader(train=False, model_config = model_config,
                                                 embeddings_config = embeddings_config,
                                                 data_config = data_config, rule_based_params=rule_based_config,
                                                 run_type = "val")
    
    print("Data Loader Creation Successful!")

    # Initialize model
    print("Creating Senseiver Object")
    model = tfm.Senseiver(
        **encoder_config,
        **decoder_config,
        **model_config,
        **embeddings_config,
        **data_config
    )
    print("Creating Senseiver Object Successful")

    path = model_config["path_working_directory"]

    print(f"Current Working Directory: {path}\n")

    # Optionally load pretrained model checkpoint
    if isinstance(checkpoints_config['load_model_num'], (int, float)):
        model_num = checkpoints_config['load_model_num']
        checkpoint_dir = f'{path}/lightning_logs/lightning_logs/version_{model_num}/checkpoints/'
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        model_loc = os.path.join(checkpoint_dir, checkpoints[-1])
        model = tfm.Senseiver.load_from_checkpoint(
            checkpoint_path = model_loc,
            strict = False,
            **encoder_config,
            **decoder_config,
            **data_config,
            **model_config,
            **embeddings_config
        )
    elif isinstance(checkpoints_config['load_model_num'], str):
        print('in')
        base_dir = Path(config['_config_dir'])
        model_loc = tfm.normalize_config_path(checkpoints_config['load_model_num'], base_dir)
        print(model_loc)
        model = tfm.Senseiver.load_from_checkpoint(
            checkpoint_path = model_loc,
            strict = False,
            **encoder_config,
            **decoder_config,
            **data_config,
            **model_config,
            **embeddings_config
        )
    else:
        model_loc = None


    def is_main_process():
        return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    
    if run_type != 'test':
        # Generate unique job ID
        job_id = int(os.getenv('SLURM_JOB_ID', datetime.now().strftime('%Y%m%d_%H%M%S')))

        # Define callbacks
        callbacks = [
            tfm.FixedModelCheckpoint(
                monitor = "train_loss",
                filename = "train-best-{epoch:02d}",
                every_n_train_steps = model_config["every_n_train_steps"],
                save_on_train_epoch_end = True
            ),
            tfm.FixedModelCheckpoint(
                every_n_train_steps = model_config["every_n_train_steps"],
                save_top_k = -1,
                filename = "train-{epoch:02d}-{step:02d}"
            ),
            tfm.FixedModelCheckpoint(
                every_n_epochs = model_config["every_n_epochs"],
                save_top_k = -1,
                filename = "train-{epoch:02d}"
            )
        ]

        # Setup loggers       
        train_logger = CSVLogger(f"{path}/logs", name="training_metrics", version=job_id, flush_logs_every_n_steps=model_config["flush_logs_every_n_steps"])
        val_logger = CSVLogger(f"{path}/logs", name="val_metrics", version=job_id, flush_logs_every_n_steps=model_config["flush_logs_every_n_steps"])
        tensorboard_logger = TensorBoardLogger(save_dir=path, name="lightning_logs", version=job_id, default_hp_metric=False)
        
        loggers = [tensorboard_logger, train_logger, val_logger]

        trainer = Trainer( max_steps = model_config['max_steps'],
                           callbacks = callbacks,
                           devices = model_config["devices"],
                           accelerator = model_config['accelerator'],
                           accumulate_grad_batches = model_config['accum_grads'],
                           log_every_n_steps = model_config['log_every_n_steps'],
                           check_val_every_n_epoch = model_config["check_val_every_n_epoch"],
                           num_nodes = model_config['num_nodes'],
                           logger = loggers,
                           strategy=DDPStrategy(find_unused_parameters=True),
                           gradient_clip_val = model_config['gradient_clip_val'],
            ) 
        
        print("Starting trainer")
        trainer.fit( model, dataloader, val_dataloader,
                        ckpt_path=model_loc
                    )
    else:
       with torch.no_grad():
            model.test(data_config, checkpoints_config, rule_based_config, test_loader=test_dataloader)
