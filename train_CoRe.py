#!/usr/bin/env python3
import numpy as np
import torch
import os
import time
import gc
import random
import shutil
import importlib.util
import inspect
import ast
import csv
import imageio.v2 as imageio
from logger import Logger
from replay_buffer import ReplayBuffer
from RRM.reward_model import RewardModel
from collections import deque
import utils
import hydra
import cv2
from omegaconf import OmegaConf
import wandb
import sys
from FRM.reward_code import FRMCode
from collections import defaultdict
from FRM.misc import file_to_string
from PRM.progress_model import ProgressRewardModel
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        utils.set_seed_everywhere(cfg.seed)

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name
            )

        # make env
        self.log_success = False
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        elif 'softgym' in cfg.env:
            self.env = utils.make_softgym_env(cfg)
        else:
            raise ValueError(f"not support the env {self.env}")

        # create agent
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        # set image size
        image_height = image_width = cfg.image_size 
        self.resize_factor = 1
        if "Rope" in cfg.env:
            image_height = image_width = 240
            self.resize_factor = 3
        elif "Water" in cfg.env:
            image_height = image_width = 360
            self.resize_factor = 2
        if "Cloth" in cfg.env:
            image_height = image_width = 360
        
        # set to 0 to make sure that rrm is 0 at the begining of the train
        self.reward_alpha = 0.0
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity), # we cannot afford to store too many images in the replay buffer.
            cfg.device,
            image_size=image_height, 
            reward_alpha=self.reward_alpha)

        # instantiating the RRM
        self.RRM = RewardModel(
            ### original PEBBLE parameters
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.RRM_lr,
            mb_size=cfg.RRM_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal,
            capacity=cfg.max_feedback,
            
            ### vlm parameters
            env_name=cfg.env,
            log_dir=self.work_dir,
            cached_label_path=cfg.cached_label_path,

            ### RRM parameters
            image_height=image_height,
            image_width=image_width,
            resize_factor=self.resize_factor,
            resnet=cfg.resnet,
            conv_kernel_sizes=cfg.conv_kernel_sizes,
            conv_strides=cfg.conv_strides,
            conv_n_channels=cfg.conv_n_channels,
        )
        

        # PRM: uses CoRe's existing VLM frame-progress labels.
        # FRM and RRM are unchanged.
        self.use_PRM = bool(getattr(cfg, "use_PRM", False))
        self.PRM_beta = float(getattr(cfg, "PRM_beta", 0.1))
        self.PRM_gamma = float(getattr(cfg, "PRM_gamma", 0.99))
        self.PRM_clip = float(getattr(cfg, "PRM_clip", 0.5))
        self.PRM_update = int(getattr(cfg, "PRM_update", 5))

        self.prm_ready = False
        self.prm_just_updated = False
        self.latest_prm_loss = 0.0

        if self.use_PRM:
            self.PRM = ProgressRewardModel(
                device=cfg.device,
                lr=float(getattr(cfg, "PRM_lr", 1e-4)),
                batch_size=int(getattr(cfg, "PRM_batch", 32)),
            )

            print(
                f"Using PRM: beta={self.PRM_beta}, "
                f"gamma={self.PRM_gamma}, "
                f"clip={self.PRM_clip}"
            )

        if cfg.reward_model_load_dir != "None":
            print("loading reward model at {}".format(cfg.reward_model_load_dir))
            self.RRM.load(cfg.reward_model_load_dir, 1000000)

        if cfg.agent_model_load_dir != "None":
            print("loading agent model at {}".format(cfg.agent_model_load_dir))
            self.agent.load(cfg.agent_model_load_dir, 1000000)

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        print(f"Using: {self.device}")

        self.image_height = image_height
        self.image_width = image_width

        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # ============================================================
        # Checkpoint / Resume
        # ============================================================

        experiment_name = str(
            getattr(cfg, "experiment", "default_experiment")
        )

        env_name = str(cfg.env).replace("/", "_")

        self.checkpoint_interval = int(
            os.environ.get(
                "CORE_CHECKPOINT_INTERVAL",
                "10000"
            )
        )

        self.checkpoint_keep = int(
            os.environ.get(
                "CORE_CHECKPOINT_KEEP",
                "2"
            )
        )        

        self.checkpoint_root = os.environ.get(
            "CORE_CHECKPOINT_DIR",
            os.path.join(
                PROJECT_DIR,
                "resume_checkpoints",
                experiment_name,
                env_name,
                f"seed{cfg.seed}"
            )
        )

        os.makedirs(
            self.checkpoint_root,
            exist_ok=True
        )

        print(
            f"[CHECKPOINT] Directory: "
            f"{self.checkpoint_root}"
        )

        print(
            f"[CHECKPOINT] Interval: "
            f"{self.checkpoint_interval} steps"
        )        

        self.save_eval_video = cfg.save_eval_video
        self.use_wandb = cfg.use_wandb

        if self.save_eval_video:
            self.save_eval_video_dir = os.path.join(self.work_dir, 'eval_videos')
            if not os.path.exists(self.save_eval_video_dir):
                os.makedirs(self.save_eval_video_dir)

        if self.use_wandb:
            group_name = f"{cfg.env}"
            wandb_name = f"{group_name}_seed{cfg.seed}"
            self.my_wandb = wandb.init(project="CoRe",
                                       group=group_name,
                                       name=wandb_name,
                                       mode="online",
                                       )
            
        # save config and prompt information
        current_file_path = os.path.dirname(os.path.realpath(__file__))
        os.system("cp {}/RRM/prompt.py {}/".format(current_file_path, self.work_dir))
        OmegaConf.save(config=self.cfg, f=f'{self.work_dir}/cfg.yaml')

        # generate FRM
        sys.path.append(f"{self.work_dir}")
        sys.path.append(f"{PROJECT_DIR}/envs/env_info")
        sys.path.append(f"{PROJECT_DIR}/envs/frm_code")
        self.FRM_gen = FRMCode(cfg, PROJECT_DIR, self.work_dir, self.RRM, llm_model=os.environ.get(
            "FRM_LLM_MODEL",
            os.environ.get("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        ), sample_num=4, once_sample_num=4)
        
        self.use_FRM_online = cfg.use_FRM_online
        # initialize the first FRM 
        if self.use_FRM_online:
            self.FRM = self.FRM_gen.FRM_generate()
            if self.FRM is None:
                raise ValueError("FRM initialization fail!")
        else:
            # load FRM from local file
            FRM_file = f"{self.cfg.env.replace('-', '_').lower()}_frm"
            self.FRM = self.FRM_gen.dynamic_import_frm(FRM_file)
            self.FRM_gen.best_frm_response = file_to_string(
                f"{PROJECT_DIR}/envs/frm_code/"
                f"{self.cfg.env.replace('-', '_').lower()}_frm.py"
            )

        # ============================================================
        # Cache Current FRM Source
        # ============================================================

        self._frm_source_cache = None

        try:
            frm_source = (
                inspect.getsourcefile(self.FRM)
                or inspect.getfile(self.FRM)
            )

            if (
                frm_source is not None
                and os.path.exists(frm_source)
            ):
                with open(
                    frm_source,
                    "r",
                    encoding="utf-8"
                ) as f:
                    self._frm_source_cache = f.read()

                print(
                    f"[FRM] Cached source from: "
                    f"{frm_source}"
                )

        except Exception as e:
            print(
                f"[FRM WARNING] Could not cache initial "
                f"FRM source: {e}"
            )

        # empty preference dataset in RAM after RRM stop update
        self.empty_pre_dataset = True

    def cleanup_old_checkpoints(self):
        checkpoint_dirs = []

        for name in os.listdir(
            self.checkpoint_root
        ):
            path = os.path.join(
                self.checkpoint_root,
                name
            )

            if (
                name.startswith("step_")
                and os.path.isdir(path)
                and not name.endswith(".tmp")
            ):
                checkpoint_dirs.append(
                    path
                )

        checkpoint_dirs.sort(
            key=lambda path: int(
                os.path.basename(path).split("_")[1]
            )
        )

        while (
            len(checkpoint_dirs)
            > self.checkpoint_keep
        ):
            old_checkpoint = (
                checkpoint_dirs.pop(0)
            )

            print(
                f"[CHECKPOINT] Removing old checkpoint: "
                f"{old_checkpoint}"
            )

            shutil.rmtree(
                old_checkpoint
            )        

    def _safe_torch_load(self, path, map_location):
        try:
            return torch.load(
                path,
                map_location=map_location,
                weights_only=False
            )
        except TypeError:
            return torch.load(
                path,
                map_location=map_location
            )        

    def _save_agent_checkpoint(self, path):
        state = {
            "actor": self.agent.actor.state_dict(),
            "critic": self.agent.critic.state_dict(),
            "critic_target": self.agent.critic_target.state_dict(),

            "actor_optimizer":
                self.agent.actor_optimizer.state_dict(),

            "critic_optimizer":
                self.agent.critic_optimizer.state_dict(),

            "log_alpha":
                self.agent.log_alpha.detach().cpu(),

            "log_alpha_optimizer":
                self.agent.log_alpha_optimizer.state_dict(),
        }

        torch.save(
            state,
            path
        )


    def _load_agent_checkpoint(self, path):
        state = self._safe_torch_load(
            path,
            self.device
        )

        self.agent.actor.load_state_dict(
            state["actor"]
        )

        self.agent.critic.load_state_dict(
            state["critic"]
        )

        self.agent.critic_target.load_state_dict(
            state["critic_target"]
        )

        self.agent.actor_optimizer.load_state_dict(
            state["actor_optimizer"]
        )

        self.agent.critic_optimizer.load_state_dict(
            state["critic_optimizer"]
        )

        self.agent.log_alpha.data.copy_(
            state["log_alpha"].to(
                self.device
            )
        )

        self.agent.log_alpha_optimizer.load_state_dict(
            state["log_alpha_optimizer"]
        )

    def _validate_frm_source(self, source, source_name="<FRM>"):
        """
        Validate that an FRM source string is syntactically valid Python
        and contains reward_function.
        """
        if source is None or not str(source).strip():
            raise ValueError(
                f"{source_name} is empty."
            )

        try:
            tree = ast.parse(
                source,
                filename=source_name
            )

        except SyntaxError as e:
            raise SyntaxError(
                f"Invalid FRM Python in {source_name}: "
                f"line {e.lineno}: {e.msg}"
            ) from e

        has_reward_function = any(
            isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.name == "reward_function"
            for node in tree.body
        )

        if not has_reward_function:
            raise ValueError(
                f"{source_name} does not contain reward_function()."
            )

        return True


    def _read_valid_frm_source(self, path):
        """
        Read and validate an FRM source file.
        """
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            source = f.read()

        self._validate_frm_source(
            source,
            source_name=path
        )

        return source
        
    def save_training_checkpoint(
        self,
        episode,
        interact_count,
        avg_train_true_return,
        RRM_learning_acc,
        vlm_acc,
    ):
        checkpoint_name = f"step_{self.step:09d}"

        checkpoint_dir = os.path.join(
            self.checkpoint_root,
            checkpoint_name
        )

        if os.path.exists(checkpoint_dir):
            print(
                f"[CHECKPOINT] Step {self.step} "
                f"already saved."
            )
            return

        temp_dir = checkpoint_dir + ".tmp"

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        os.makedirs(
            temp_dir,
            exist_ok=True
        )

        print("")
        print(
            f"[CHECKPOINT] Saving checkpoint "
            f"at step {self.step}..."
        )

        # ========================================================
        # Save SAC
        # ========================================================

        agent_path = os.path.join(
            temp_dir,
            "agent.pt"
        )

        self._save_agent_checkpoint(
            agent_path
        )

        # ========================================================
        # Save Replay Buffer
        # ========================================================

        replay_dir = os.path.join(
            temp_dir,
            "replay_buffer"
        )

        if not hasattr(
            self.replay_buffer,
            "save"
        ):
            raise RuntimeError(
                "ReplayBuffer needs a save() function "
                "before checkpointing can be used."
            )

        self.replay_buffer.save(
            replay_dir
        )

        # ========================================================
        # Save RRM
        # ========================================================

        rrm_dir = os.path.join(
            temp_dir,
            "rrm"
        )

        if not hasattr(
            self.RRM,
            "save_checkpoint"
        ):
            raise RuntimeError(
                "RRM needs save_checkpoint() "
                "before checkpointing can be used."
            )

        self.RRM.save_checkpoint(
            rrm_dir
        )

        # ========================================================
        # Save PRM
        # ========================================================

        if self.use_PRM:
            prm_state = {
                "model":
                    self.PRM.state_dict(),

                "optimizer":
                    self.PRM.opt.state_dict(),

                "prm_ready":
                    self.prm_ready,

                "prm_just_updated":
                    self.prm_just_updated,

                "latest_prm_loss":
                    self.latest_prm_loss,
            }

            prm_path = os.path.join(
                temp_dir,
                "prm.pt"
            )

            torch.save(
                prm_state,
                prm_path
            )

        # ========================================================
        # Save Current FRM
        # ========================================================

        frm_checkpoint_path = os.path.join(
            temp_dir,
            "current_frm.py"
        )

        frm_source_path = None
        frm_source_code = None

        try:
            frm_source_path = (
                inspect.getsourcefile(self.FRM)
                or inspect.getfile(self.FRM)
            )

            if (
                frm_source_path is not None
                and os.path.exists(frm_source_path)
            ):
                try:
                    frm_source_code = self._read_valid_frm_source(
                        frm_source_path
                    )

                    print(
                        f"[CHECKPOINT] Current FRM source validated: "
                        f"{frm_source_path}"
                    )

                except Exception as e:
                    print(
                        "[CHECKPOINT WARNING] Current FRM source "
                        f"is invalid: {e}"
                    )

                    frm_source_code = None

        except Exception as e:
            print(
                "[CHECKPOINT WARNING] Could not determine "
                f"FRM source path: {e}"
            )

        if frm_source_code is None:

            cached_source = getattr(
                self,
                "_frm_source_cache",
                None
            )

            if cached_source is not None:

                try:
                    self._validate_frm_source(
                        cached_source,
                        source_name="cached_FRM"
                    )

                    frm_source_code = cached_source

                    print(
                        "[CHECKPOINT] Using last known-good "
                        "cached FRM source."
                    )

                except Exception as e:
                    print(
                        "[CHECKPOINT WARNING] Cached FRM is "
                        f"also invalid: {e}"
                    )

                    frm_source_code = None

        if frm_source_code is None:
            raise RuntimeError(
                "Cannot save checkpoint because no valid FRM "
                "source is available."
            )

        self._validate_frm_source(
            frm_source_code,
            source_name="checkpoint_FRM"
        )

        with open(
            frm_checkpoint_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(
                frm_source_code
            )

        self._frm_source_cache = frm_source_code

        print(
            "[CHECKPOINT] Valid current FRM saved successfully."
        )

        # ========================================================
        # Save Main Training State
        # ========================================================

        training_state = {
            "step":
                self.step,

            "episode":
                episode,

            "total_feedback":
                self.total_feedback,

            "labeled_feedback":
                self.labeled_feedback,

            "reward_alpha":
                self.reward_alpha,

            "interact_count":
                interact_count,

            "RRM_learning_acc":
                RRM_learning_acc,

            "vlm_acc":
                vlm_acc,

            "avg_train_true_return":
                list(avg_train_true_return),

            "frm_metric_dict":
                dict(self.FRM_gen.metric_dict),

            "frm_iter":
                int(self.FRM_gen.iter),

            "frm_best_response":
                self.FRM_gen.best_frm_response,            

            "numpy_rng":
                np.random.get_state(),

            "python_rng":
                random.getstate(),

            "torch_rng":
                torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            training_state["cuda_rng"] = (
                torch.cuda.get_rng_state_all()
            )

        training_state_path = os.path.join(
            temp_dir,
            "training_state.pt"
        )

        torch.save(
            training_state,
            training_state_path
        )

        # ========================================================
        # Mark Checkpoint Complete
        # ========================================================

        os.rename(
            temp_dir,
            checkpoint_dir
        )

        # ========================================================
        # Update latest.txt
        # ========================================================

        latest_tmp = os.path.join(
            self.checkpoint_root,
            "latest.txt.tmp"
        )

        latest_file = os.path.join(
            self.checkpoint_root,
            "latest.txt"
        )

        with open(
            latest_tmp,
            "w"
        ) as f:
            f.write(
                checkpoint_name
            )

        os.replace(
            latest_tmp,
            latest_file
        )

        # ========================================================
        # Remove Old Checkpoints
        # ========================================================

        self.cleanup_old_checkpoints()

        print(
            "[CHECKPOINT] Saved successfully."
        )

        print(
            f"[CHECKPOINT] {checkpoint_dir}"
        )

        print("")


    def load_latest_checkpoint(self):
        latest_file = os.path.join(
            self.checkpoint_root,
            "latest.txt"
        )

        if not os.path.exists(latest_file):
            print(
                "[CHECKPOINT] No previous checkpoint found."
            )

            print(
                "[CHECKPOINT] Starting training from step 0."
            )

            return None

        with open(
            latest_file,
            "r"
        ) as f:
            checkpoint_name = (
                f.read().strip()
            )

        checkpoint_dir = os.path.join(
            self.checkpoint_root,
            checkpoint_name
        )

        state_path = os.path.join(
            checkpoint_dir,
            "training_state.pt"
        )

        if not os.path.exists(state_path):
            raise RuntimeError(
                f"Invalid checkpoint: "
                f"{checkpoint_dir}"
            )

        print("")
        print(
            "============================================"
        )

        print(
            "[RESUME] Loading checkpoint:"
        )

        print(
            checkpoint_dir
        )

        print(
            "============================================"
        )

        state = self._safe_torch_load(
            state_path,
            "cpu"
        )

        # ========================================================
        # Load SAC
        # ========================================================

        agent_path = os.path.join(
            checkpoint_dir,
            "agent.pt"
        )

        self._load_agent_checkpoint(
            agent_path
        )

        # ========================================================
        # Load Replay Buffer
        # ========================================================

        replay_dir = os.path.join(
            checkpoint_dir,
            "replay_buffer"
        )

        if not hasattr(
            self.replay_buffer,
            "load"
        ):
            raise RuntimeError(
                "ReplayBuffer needs a load() function."
            )

        self.replay_buffer.load(
            replay_dir
        )

        # ========================================================
        # Load RRM
        # ========================================================

        rrm_dir = os.path.join(
            checkpoint_dir,
            "rrm"
        )

        if not hasattr(
            self.RRM,
            "load_checkpoint"
        ):
            raise RuntimeError(
                "RRM needs load_checkpoint()."
            )

        self.RRM.load_checkpoint(
            rrm_dir
        )

        # ========================================================
        # Load PRM
        # ========================================================

        if self.use_PRM:
            prm_path = os.path.join(
                checkpoint_dir,
                "prm.pt"
            )

            prm_state = self._safe_torch_load(
                prm_path,
                self.device
            )

            self.PRM.load_state_dict(
                prm_state["model"]
            )

            self.PRM.opt.load_state_dict(
                prm_state["optimizer"]
            )

            self.prm_ready = bool(
                prm_state["prm_ready"]
            )

            self.prm_just_updated = bool(
                prm_state["prm_just_updated"]
            )

            self.latest_prm_loss = float(
                prm_state["latest_prm_loss"]
            )

        # ========================================================
        # Restore Current FRM
        # ========================================================

        frm_checkpoint_path = os.path.join(
            checkpoint_dir,
            "current_frm.py"
        )

        if os.path.exists(frm_checkpoint_path):

            # ----------------------------------------------------
            # Validate checkpoint FRM before importing
            # ----------------------------------------------------

            with open(
                frm_checkpoint_path,
                "r",
                encoding="utf-8"
            ) as f:
                checkpoint_frm_source = f.read()

            self._validate_frm_source(
                checkpoint_frm_source,
                source_name=frm_checkpoint_path
            )

            print(
                "[RESUME] Checkpoint FRM syntax validated."
            )

            module_name = (
                f"_checkpoint_frm_{int(state['step'])}"
            )

            spec = importlib.util.spec_from_file_location(
                module_name,
                frm_checkpoint_path
            )

            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Could not load checkpoint FRM: "
                    f"{frm_checkpoint_path}"
                )

            frm_module = importlib.util.module_from_spec(
                spec
            )

            spec.loader.exec_module(
                frm_module
            )

            if not hasattr(frm_module, "reward_function"):
                raise RuntimeError(
                    "Checkpoint FRM does not contain "
                    "reward_function."
                )

            self.FRM = frm_module.reward_function

            # ====================================================
            # Cache restored FRM source
            # ====================================================

            try:
                with open(
                    frm_checkpoint_path,
                    "r",
                    encoding="utf-8"
                ) as f:
                    self._frm_source_cache = f.read()

                print(
                    "[RESUME] FRM source cached in memory."
                )

            except Exception as e:
                print(
                    f"[RESUME WARNING] Could not cache "
                    f"FRM source: {e}"
                )

            self.FRM_gen.iter = int(
                state.get(
                    "frm_iter",
                    self.FRM_gen.iter
                )
            )

            self.FRM_gen.best_frm_response = (
                state.get(
                    "frm_best_response",
                    self.FRM_gen.best_frm_response
                )
            )

            print(
                f"[RESUME] Current FRM restored. "
                f"FRM iteration={self.FRM_gen.iter}"
            )

        else:
            print(
                "[RESUME] WARNING: old checkpoint does not "
                "contain current_frm.py. Using initialized FRM."
            )
        
        # ========================================================
        # Restore Counters
        # ========================================================

        self.step = int(
            state["step"]
        )

        self.total_feedback = int(
            state["total_feedback"]
        )

        self.labeled_feedback = int(
            state["labeled_feedback"]
        )

        self.reward_alpha = float(
            state["reward_alpha"]
        )

        self.replay_buffer.reward_alpha = (
            self.reward_alpha
        )

        # ========================================================
        # Restore FRM History
        # ========================================================

        self.FRM_gen.metric_dict = state.get(
            "frm_metric_dict",
            {}
        )

        # ========================================================
        # Restore RNG States
        # ========================================================

        np.random.set_state(
            state["numpy_rng"]
        )

        random.setstate(
            state["python_rng"]
        )

        torch.set_rng_state(
            state["torch_rng"]
        )

        if (
            torch.cuda.is_available()
            and "cuda_rng" in state
        ):
            torch.cuda.set_rng_state_all(
                state["cuda_rng"]
            )

        print(
            f"[RESUME] Step: "
            f"{self.step}"
        )

        print(
            f"[RESUME] Episode: "
            f"{state['episode']}"
        )

        print(
            f"[RESUME] Feedback: "
            f"{self.total_feedback}/"
            f"{self.cfg.max_feedback}"
        )

        print(
            f"[RESUME] Replay size: "
            f"{len(self.replay_buffer)}"
        )

        print(
            "[RESUME] Checkpoint loaded successfully."
        )

        print(
            "============================================"
        )

        print("")

        return state    

    def evaluate(self, train_episode):
        average_true_episode_reward = 0.0
        success_rate = 0.0

        # ============================================================
        # Store all individual evaluation episodes for this checkpoint
        # ============================================================
        eval_rows = []

        for eval_episode in range(self.cfg.num_eval_episodes):
            print(
                "evaluating episode {}".format(
                    eval_episode
                )
            )

            images = []

            obs = self.env.reset()

            if "metaworld" in self.cfg.env:
                obs = obs[0]

                # ========================================================
                # Diagnostic: check evaluation reset variation
                # ========================================================
                base_env = self.env

                while True:
                    if hasattr(base_env, "env"):
                        base_env = base_env.env
                    elif hasattr(base_env, "_wrapped_env"):
                        base_env = base_env._wrapped_env
                    elif hasattr(base_env, "wrapped_env"):
                        base_env = base_env.wrapped_env
                    else:
                        break

                print(
                    f"[EVAL ENV CHECK] "
                    f"cfg.env={self.cfg.env} | "
                    f"class={base_env.__class__.__name__}"
                )

                if hasattr(base_env, "_random_reset_space"):
                    print(
                        f"[EVAL RESET SPACE] "
                        f"low={np.round(base_env._random_reset_space.low, 5)} | "
                        f"high={np.round(base_env._random_reset_space.high, 5)}"
                    )

                if hasattr(base_env, "obj_init_pos"):
                    print(
                        f"[EVAL OBJECT] "
                        f"obj_init_pos={np.round(base_env.obj_init_pos, 5)}"
                    )

                if hasattr(base_env, "_target_pos"):
                    print(
                        f"[EVAL TARGET] "
                        f"target_pos={np.round(base_env._target_pos, 5)}"
                    )

                print(
                    f"[EVAL RESET CHECK] "
                    f"eval_episode={eval_episode} | "
                    f"obs={np.round(obs, 5)}"
                )

                print(
                    f"[EVAL OBS INFO] "
                    f"eval_episode={eval_episode} | "
                    f"obs_shape={obs.shape} | "
                    f"obs_sum={np.sum(obs):.8f}"
                )

                if hasattr(base_env, "_last_rand_vec"):
                    print(
                        f"[EVAL RESET CHECK] "
                        f"eval_episode={eval_episode} | "
                        f"rand_vec={np.round(base_env._last_rand_vec, 5)}"
                    )

            done = False

            true_episode_reward = 0.0

            eval_step_count = 0

            if self.log_success:
                episode_success = 0.0

            # ========================================================
            # Run one complete evaluation episode
            # ========================================================
            while not done:

                with utils.eval_mode(self.agent):
                    action = self.agent.act(
                        obs,
                        sample=False
                    )

                try:
                    obs, reward, done, extra = self.env.step(
                        action
                    )

                except:
                    (
                        obs,
                        reward,
                        terminated,
                        truncated,
                        extra
                    ) = self.env.step(
                        action
                    )

                    done = terminated or truncated
                eval_step_count += 1                    

                # ====================================================
                # True environment reward
                # ====================================================
                if "Cloth" in self.cfg.env:
                    true_episode_reward = reward

                else:
                    true_episode_reward += reward

                # ====================================================
                # Success
                # ====================================================
                if self.log_success:
                    episode_success = max(
                        episode_success,
                        float(extra["success"])
                    )

                # ====================================================
                # Evaluation video
                # ====================================================
                if self.save_eval_video:

                    if "metaworld" in self.cfg.env:

                        rgb_image = self.env.render()

                        rgb_image = rgb_image[::-1, :, :]

                        if (
                            "drawer" in self.cfg.env
                            or "sweep" in self.cfg.env
                        ):
                            rgb_image = rgb_image[
                                100:400,
                                100:400,
                                :
                            ]

                    else:

                        rgb_image = self.env.render(
                            mode="rgb_array"
                        )

                    if "softgym" not in self.cfg.env:
                        images.append(
                            rgb_image
                        )

            # ========================================================
            # SoftGym evaluation video frames
            # ========================================================
            if self.save_eval_video:

                if "softgym" in self.cfg.env:
                    images = self.env.video_frames

            # ========================================================
            # Save evaluation video
            # ========================================================
            if self.save_eval_video:

                if self.log_success:

                    save_video_path = os.path.join(
                        self.save_eval_video_dir,
                        (
                            "step{:07}_episode{:02}_"
                            "success{}_{}.mp4"
                        ).format(
                            self.step,
                            eval_episode,
                            episode_success,
                            round(
                                float(true_episode_reward),
                                2
                            )
                        )
                    )

                else:

                    save_video_path = os.path.join(
                        self.save_eval_video_dir,
                        "step{:07}_episode{:02}_{}.mp4".format(
                            self.step,
                            eval_episode,
                            round(
                                float(true_episode_reward),
                                2
                            )
                        )
                    )

                utils.save_list_as_mp4(
                    images,
                    save_video_path
                )

            # ========================================================
            # Convert individual success to percentage
            #
            # Successful episode   -> 100.0
            # Unsuccessful episode ->   0.0
            # ========================================================
            if self.log_success:

                if episode_success > 0.0:
                    individual_success_rate = 100.0

                else:
                    individual_success_rate = 0.0

                success_rate += individual_success_rate

            else:

                individual_success_rate = float("nan")

            # ========================================================
            # Convert reward to standard Python float
            # ========================================================
            individual_true_episode_reward = float(
                true_episode_reward
            )
            
            print(
                f"[EVAL RESULT] "
                f"eval_episode={eval_episode} | "
                f"steps={eval_step_count} | "
                f"success={individual_success_rate:.1f} | "
                f"reward={individual_true_episode_reward:.6f}"
            )            

            average_true_episode_reward += (
                individual_true_episode_reward
            )

            # ========================================================
            # Save individual evaluation rows only for MetaWorld.
            #
            # SoftGym will save only the aggregate mean later.
            # ========================================================
            if self.log_success:
                eval_rows.append(
                    {
                        "episode": int(train_episode),
                        "eval_episode": int(eval_episode),
                        "step": int(self.step),
                        "success_rate": individual_success_rate,
                        "true_episode_reward":
                            individual_true_episode_reward,
                    }
                )
                
            # ========================================================
            # Print individual evaluation result
            # ========================================================
            if self.log_success:

                print(
                    f"[EVAL EPISODE] "
                    f"episode={train_episode} | "
                    f"eval_episode={eval_episode} | "
                    f"step={self.step} | "
                    f"success_rate="
                    f"{individual_success_rate:.1f} | "
                    f"true_episode_reward="
                    f"{individual_true_episode_reward:.4f}"
                )

            else:

                print(
                    f"[EVAL EPISODE] "
                    f"episode={train_episode} | "
                    f"eval_episode={eval_episode} | "
                    f"step={self.step} | "
                    f"true_episode_reward="
                    f"{individual_true_episode_reward:.4f}"
                )

        # ============================================================
        # Calculate aggregate result across all evaluation episodes
        # ============================================================
        average_true_episode_reward /= (
            self.cfg.num_eval_episodes
        )

        if self.log_success:
            success_rate /= (
                self.cfg.num_eval_episodes
            )
        # ============================================================
        # Save evaluation CSV
        #
        # MetaWorld:
        #   Save individual evaluation episodes and success rate.
        #
        # SoftGym:
        #   Save only ONE aggregate row containing the mean reward.
        # ============================================================

        eval_csv_path = os.path.join(
            self.work_dir,
            "eval.csv"
        )

        eval_csv_exists = (
            os.path.exists(eval_csv_path)
            and os.path.getsize(eval_csv_path) > 0
        )

        # ============================================================
        # MetaWorld
        # ============================================================
        if self.log_success:

            fieldnames = [
                "episode",
                "eval_episode",
                "step",
                "success_rate",
                "true_episode_reward",
            ]

            with open(
                eval_csv_path,
                "a",
                newline=""
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames
                )

                if not eval_csv_exists:
                    writer.writeheader()

                writer.writerows(
                    eval_rows
                )

        # ============================================================
        # SoftGym
        # ============================================================
        else:

            fieldnames = [
                "episode",
                "step",
                "true_episode_reward",
            ]

            aggregate_row = {
                "episode": int(train_episode),
                "step": int(self.step),
                "true_episode_reward":
                    float(average_true_episode_reward),
            }

            with open(
                eval_csv_path,
                "a",
                newline=""
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames
                )

                if not eval_csv_exists:
                    writer.writeheader()

                writer.writerow(
                    aggregate_row
                )

        # ============================================================
        # Print aggregate evaluation summary
        # ============================================================
        if self.log_success:

            print(
                ""
            )

            print(
                f"[EVAL SUMMARY] "
                f"episode={train_episode} | "
                f"step={self.step} | "
                f"success_rate={success_rate:.4f} | "
                f"true_episode_reward="
                f"{average_true_episode_reward:.4f}"
            )

            print(
                ""
            )

        else:

            print(
                ""
            )

            print(
                f"[EVAL SUMMARY] "
                f"episode={train_episode} | "
                f"step={self.step} | "
                f"true_episode_reward="
                f"{average_true_episode_reward:.4f}"
            )

            print(
                ""
            )

        # ============================================================
        # Keep aggregate WandB logging unchanged
        # ============================================================
        if (
            self.log_success
            and self.use_wandb
        ):

            self.my_wandb.log(
                {
                    f"{self.cfg.env}":
                        success_rate
                },
                step=self.step
            )

    # ============================================================
    # Final Demo GIF
    # Works with SoftGym, MetaWorld, and other CoRe environments
    # ============================================================

    def save_final_demo_gif(self, num_episodes=3, fps=2):

        save_dir = os.path.join(
            self.work_dir,
            "final_demo"
        )

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        print("")
        print("============================================================")
        print("[FINAL DEMO] Generating final animated GIF...")
        print("============================================================")

        env_name = str(self.cfg.env).lower()

        best_frames = None
        best_score = -float("inf")
        best_success = 0
        best_reward = -float("inf")

        # ========================================================
        # Check whether env.step() supports SoftGym continuous
        # video recording.
        #
        # This makes the function generic instead of checking
        # specific tasks such as ClothFoldDiagonal.
        # ========================================================

        supports_continuous_video = False

        try:

            step_signature = inspect.signature(
                self.env.step
            )

            if (
                "record_continuous_video"
                in step_signature.parameters
            ):

                supports_continuous_video = True

        except Exception:

            supports_continuous_video = False

        print(
            "[FINAL DEMO] Continuous simulator recording: "
            f"{supports_continuous_video}"
        )

        # ========================================================
        # Helper: convert any rendered image into a clean
        # 300 x 300 RGB uint8 frame
        # ========================================================

        def prepare_frame(frame):

            if frame is None:
                return None

            if isinstance(frame, (list, tuple)):

                if len(frame) == 0:
                    return None

                frame = frame[0]

            frame = np.asarray(frame)

            # Some renderers may return:
            # [1, H, W, C]
            if frame.ndim == 4:
                frame = frame[0]

            if frame.ndim != 3:
                return None

            # Remove alpha channel if present
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]

            # MetaWorld/MuJoCo offscreen images in this project
            # may need vertical flipping.
            if "metaworld" in env_name:
                frame = frame[::-1, :, :]

            # Convert floating-point images to uint8 if needed
            if frame.dtype != np.uint8:

                frame = frame.astype(
                    np.float32
                )

                if frame.max() <= 1.0:
                    frame = frame * 255.0

                frame = np.clip(
                    frame,
                    0,
                    255
                ).astype(np.uint8)

            frame = cv2.resize(
                frame,
                (300, 300)
            )

            return frame.copy()

        # ========================================================
        # Helper: render current environment
        #
        # Try several render APIs so this works across tasks.
        # ========================================================

        def capture_current_frame():

            frame = None

            # ----------------------------------------------------
            # Attempt 1
            # SoftGym style
            # ----------------------------------------------------

            try:

                frame = self.env.render(
                    mode="rgb_array",
                    hide_picker=True
                )

            except Exception:
                frame = None

            # ----------------------------------------------------
            # Attempt 2
            # Standard Gym / SoftGym style
            # ----------------------------------------------------

            if frame is None:

                try:

                    frame = self.env.render(
                        mode="rgb_array"
                    )

                except Exception:
                    frame = None

            # ----------------------------------------------------
            # Attempt 3
            # MetaWorld / modern Gym style
            # ----------------------------------------------------

            if frame is None:

                try:

                    frame = self.env.render()

                except Exception:
                    frame = None

            return prepare_frame(
                frame
            )

        # ========================================================
        # Generate evaluation/demo episodes
        # ========================================================

        for ep in range(num_episodes):

            # ----------------------------------------------------
            # Reset environment
            # ----------------------------------------------------

            reset_output = self.env.reset()

            # Gymnasium can return:
            # (observation, info)
            if (
                isinstance(reset_output, tuple)
                and len(reset_output) == 2
            ):

                obs = reset_output[0]

            else:

                obs = reset_output

            done = False

            frames = []

            episode_reward = 0.0
            episode_success = 0

            policy_step_count = 0

            # ----------------------------------------------------
            # Prevent an accidental infinite episode
            # ----------------------------------------------------

            max_demo_steps = None

            for attr_name in [
                "_max_episode_steps",
                "horizon",
                "max_episode_steps"
            ]:

                if hasattr(self.env, attr_name):

                    try:

                        value = int(
                            getattr(
                                self.env,
                                attr_name
                            )
                        )

                        if value > 0:
                            max_demo_steps = value
                            break

                    except Exception:
                        pass

            if max_demo_steps is None:

                # Safety limit only.
                # Normal environments should finish before this.
                max_demo_steps = 1000

            # ----------------------------------------------------
            # Capture reset / initial frame
            # ----------------------------------------------------

            try:

                initial_frame = capture_current_frame()

                if initial_frame is not None:

                    frames.append(
                        initial_frame
                    )

            except Exception as e:

                print(
                    "[FINAL DEMO] Initial frame warning: "
                    f"{e}"
                )

            # ====================================================
            # Full policy rollout
            # ====================================================

            while (
                not done
                and policy_step_count < max_demo_steps
            ):

                # ------------------------------------------------
                # Get deterministic action from trained policy
                # ------------------------------------------------

                with utils.eval_mode(self.agent):

                    action = self.agent.act(
                        obs,
                        sample=False
                    )

                # ------------------------------------------------
                # Environment step
                #
                # SoftGym:
                # request continuous simulator frames.
                #
                # MetaWorld/other envs:
                # normal step.
                # ------------------------------------------------

                if supports_continuous_video:

                    step_output = self.env.step(
                        action,
                        record_continuous_video=True,
                        img_size=300
                    )

                else:

                    step_output = self.env.step(
                        action
                    )

                # ------------------------------------------------
                # Support both Gym and Gymnasium APIs
                # ------------------------------------------------

                if len(step_output) == 4:

                    (
                        next_obs,
                        reward,
                        done,
                        info
                    ) = step_output

                elif len(step_output) == 5:

                    (
                        next_obs,
                        reward,
                        terminated,
                        truncated,
                        info
                    ) = step_output

                    done = bool(
                        terminated or truncated
                    )

                else:

                    raise RuntimeError(
                        "[FINAL DEMO] env.step() returned "
                        f"{len(step_output)} values. "
                        "Expected 4 or 5."
                    )

                # ------------------------------------------------
                # Reward
                # ------------------------------------------------

                try:

                    episode_reward += float(
                        reward
                    )

                except Exception:

                    episode_reward += float(
                        np.asarray(reward).mean()
                    )

                # ------------------------------------------------
                # Detect success using common environment keys
                # ------------------------------------------------

                if isinstance(info, dict):

                    for success_key in [
                        "success",
                        "is_success",
                        "task_success"
                    ]:

                        if success_key in info:

                            try:

                                success_value = float(
                                    info[success_key]
                                )

                                episode_success = max(
                                    episode_success,
                                    success_value
                                )

                            except Exception:
                                pass

                # =================================================
                # FRAME COLLECTION
                # =================================================

                frames_added = 0

                # ------------------------------------------------
                # SoftGym continuous simulator frames
                #
                # This is the important fix for environments where
                # one policy action contains many physics frames.
                # ------------------------------------------------

                if isinstance(info, dict):

                    continuous_frames = info.get(
                        "flex_env_recorded_frames",
                        None
                    )

                    if continuous_frames is not None:

                        for raw_frame in continuous_frames:

                            frame = prepare_frame(
                                raw_frame
                            )

                            if frame is not None:

                                frames.append(
                                    frame
                                )

                                frames_added += 1

                # ------------------------------------------------
                # Fallback:
                # capture one frame after EVERY policy action.
                #
                # Important:
                # NO "step_count % 4" condition anymore.
                # ------------------------------------------------

                if frames_added == 0:

                    try:

                        frame = capture_current_frame()

                        if frame is not None:

                            frames.append(
                                frame
                            )

                            frames_added += 1

                    except Exception as e:

                        print(
                            "[FINAL DEMO] Frame capture "
                            f"warning: {e}"
                        )

                obs = next_obs

                policy_step_count += 1

            # ====================================================
            # Episode statistics
            # ====================================================

            print(
                f"[FINAL DEMO] Episode {ep} | "
                f"Policy Steps: {policy_step_count} | "
                f"Success: {episode_success} | "
                f"Reward: {episode_reward:.3f} | "
                f"GIF Frames: {len(frames)}"
            )

            # ====================================================
            # Choose best demo
            #
            # Prefer successful episode.
            # Reward breaks ties.
            # ====================================================

            score = (
                float(episode_success) * 1000000.0
                + float(episode_reward)
            )

            if score > best_score:

                best_score = score

                best_frames = [
                    frame.copy()
                    for frame in frames
                ]

                best_success = episode_success
                best_reward = episode_reward

        # ========================================================
        # Make sure we actually have an animation
        # ========================================================

        if best_frames is None:

            print("")
            print(
                "[FINAL DEMO] ERROR: "
                "No frames were captured."
            )
            print("")

            return

        if len(best_frames) < 2:

            print("")
            print(
                "[FINAL DEMO] ERROR: Only "
                f"{len(best_frames)} frame was captured."
            )

            print(
                "[FINAL DEMO] GIF was NOT saved because "
                "it would only be a static image."
            )

            print("")

            return

        # ========================================================
        # Save animated GIF
        # ========================================================

        gif_path = os.path.join(
            save_dir,
            (
                f"final_demo_step{self.step}_"
                f"success{int(best_success)}_"
                f"reward{best_reward:.2f}.gif"
            )
        )

        # Pillow/imageio expects milliseconds per frame
        frame_duration = int(
            1000 / fps
        )

        imageio.mimsave(
            gif_path,
            best_frames,
            duration=frame_duration,
            loop=0
        )

        print("")
        print("============================================================")
        print("[FINAL DEMO] Animated GIF saved successfully")
        print("============================================================")

        print(
            f"[FINAL DEMO] Path: {gif_path}"
        )

        print(
            f"[FINAL DEMO] Total GIF frames: "
            f"{len(best_frames)}"
        )

        print(
            f"[FINAL DEMO] FPS: {fps}"
        )

        print(
            f"[FINAL DEMO] Success: "
            f"{best_success}"
        )

        print(
            f"[FINAL DEMO] Reward: "
            f"{best_reward:.3f}"
        )

        print("============================================================")
        print("")


    def learn_RRM(self, first_flag=0):
        # get feedbacks
        query_label_timer = time.time()
        labeled_queries = self.RRM.uniform_sampling()
        self.total_feedback += self.RRM.mb_size
        self.labeled_feedback += labeled_queries

        print(f'Query label time: {time.time() - query_label_timer}')
        self.logger.log(
            'train_time/query_label',
            time.time() - query_label_timer,
            self.step
        )

        # Clear unused CPU/GPU cached memory before RRM training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_acc = 0
        RRM_learning_timer = time.time()

        if self.labeled_feedback > 0:
            # update RRM
            for _ in range(self.cfg.RRM_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    raise NotImplementedError
                else:
                    self.RRM.train()

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    train_acc = self.RRM.train_reward()

                total_acc = np.mean(train_acc)

                if total_acc > 0.97:
                    break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.log(
            'train_time/RRM_learning',
            time.time() - RRM_learning_timer,
            self.step
        )

        print(
            f"RRM is updated! ACC: {total_acc:.3f} "
            f"Queried/Total: {self.labeled_feedback}/{self.total_feedback}"
        )

        print(
            f'RRM learning time: {time.time() - RRM_learning_timer}'
        )

        return total_acc, self.RRM.vlm_label_acc

    def relabel_with_RRM(self):
        self.RRM.eval()
        self.replay_buffer.relabel_with_predictor(self.RRM, self.step)
        self.RRM.train()

    def learn_PRM(self):
        if not self.use_PRM:
            return 0.0

        loss = self.PRM.train_from_rrm(
            self.RRM,
            updates=self.PRM_update
        )

        self.prm_ready = True
        self.prm_just_updated = True

        # Store latest loss.
        # Do NOT directly call self.logger.log here.
        self.latest_prm_loss = float(loss)

        print(
            f"[PRM] Updated at step {self.step} | "
            f"LOSS: {self.latest_prm_loss:.5f}"
        )

        if self.use_wandb:
            self.my_wandb.log(
                {"train/PRM_loss": self.latest_prm_loss},
                step=self.step
            )

        return loss

    def relabel_with_PRM(self):
        # Re-add PRM after CoRe FRM/RRM relabeling.
        if not self.use_PRM or not self.prm_ready:
            return

        size = (
            self.replay_buffer.capacity
            if self.replay_buffer.full
            else self.replay_buffer.idx
        )

        if size == 0:
            return

        print(
            f"[PRM RELABEL] Starting PRM relabel for "
            f"{size} samples..."
        )

        # ----------------------------------------------------
        # Predict progress in small batches.
        # This avoids sending the full replay buffer to PRM
        # at the same time.
        # ----------------------------------------------------
        batch_size = 128

        progress = np.zeros(
            size,
            dtype=np.float32
        )

        for start in range(0, size, batch_size):
            end = min(
                start + batch_size,
                size
            )

            batch_images = self.replay_buffer.images[start:end]

            batch_progress = self.PRM.predict_batch(
                batch_images
            )

            batch_progress = np.asarray(
                batch_progress,
                dtype=np.float32
            ).reshape(-1)

            progress[start:end] = batch_progress

            if end % 1024 == 0 or end == size:
                print(
                    f"[PRM RELABEL] "
                    f"{end}/{size} samples completed"
                )

        # ----------------------------------------------------
        # Calculate PRM reward
        # ----------------------------------------------------
        reward_prm = np.zeros(
            size,
            dtype=np.float32
        )

        for i in range(size):
            if (
                i == 0
                or self.replay_buffer.not_dones[i - 1, 0] == 0
            ):
                prev = 0.0
            else:
                prev = progress[i - 1]

            reward_prm[i] = (
                self.PRM_gamma * progress[i]
                - prev
            )

        reward_prm = np.clip(
            reward_prm,
            -self.PRM_clip,
            self.PRM_clip
        )

        # ----------------------------------------------------
        # Add weighted PRM reward to replay buffer
        # ----------------------------------------------------
        self.replay_buffer.rewards[:size] += (
            self.PRM_beta
            * reward_prm.reshape(-1, 1)
        )

        print(
            f"[PRM RELABEL] Finished relabeling "
            f"{size} samples."
        )

        # ----------------------------------------------------
        # Clean temporary memory
        # ----------------------------------------------------
        del progress
        del reward_prm

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self):
        model_save_dir = os.path.join(self.work_dir, "models")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        episode, episode_reward, done = 0, 0, True
        episode_end_reward = 0
        episode_reward_frm, episode_reward_rrm, episode_reward_prm = 0, 0, 0
        true_episode_reward = 0
        if self.log_success:
            episode_success = 0

        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10)

        start_time = time.time()
        # for initialize wandb when step=0
        if self.use_wandb:
            self.my_wandb.log({f"{self.cfg.env}": 0}, step=0)

        interact_count = 0
        RRM_learning_acc = 0
        vlm_acc = 0

        # ============================================================
        # Resume Training
        # ============================================================

        resume_state = self.load_latest_checkpoint()

        if resume_state is not None:
            episode = int(
                resume_state["episode"]
            )

            interact_count = int(
                resume_state["interact_count"]
            )

            RRM_learning_acc = float(
                resume_state["RRM_learning_acc"]
            )

            vlm_acc = float(
                resume_state["vlm_acc"]
            )

            avg_train_true_return.clear()

            for value in resume_state.get(
                "avg_train_true_return",
                []
            ):
                avg_train_true_return.append(
                    value
                )

            # ========================================================
            # Start a Fresh Environment Episode After Resume
            # ========================================================

            obs = self.env.reset()

            if "metaworld" in self.cfg.env:
                obs = obs[0]

            # Restore goal information used by FRM.
            if "metaworld" in self.cfg.env:
                self.FRM_gen.target_pos = self.env.target_pos_frm
            elif "Water" in self.cfg.env:
                self.FRM_gen.target_pos = 1.2

            done = False

            episode_reward = 0
            episode_reward_frm = 0
            episode_reward_rrm = 0
            episode_reward_prm = 0

            true_episode_reward = 0

            if self.log_success:
                episode_success = 0

            episode_step = 0
            prev_progress = 0.0

            episode += 1

            self.logger.log(
                'train/episode',
                episode,
                self.step
            )

            step_frm_dict = []

            print(
                f"[RESUME] Starting fresh episode "
                f"{episode} from training step {self.step}"
            )

        # ============================================================
        # Next Checkpoint Step
        # ============================================================

        next_checkpoint_step = (
            (
                self.step
                // self.checkpoint_interval
            )
            + 1
        ) * self.checkpoint_interval

        # ============================================================
        # Main Training Loop
        # ============================================================

        while self.step <= self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    # record time
                    self.logger.log('train_time/duration', time.time() - start_time, self.step)
                    start_time = time.time()

                    # logging for frm component
                    episode_reward_dict = defaultdict(float)
                    for d in step_frm_dict:
                        for key, value in d.items():
                            episode_reward_dict[key] += value
                    episode_reward_dict = dict(episode_reward_dict)
               
                    # FRM
                    if self.step <= self.cfg.FRM_align_max_steps:
                        # FRM component logging for policy feedback
                        if len(self.FRM_gen.metric_dict) == 0:
                            if "metaworld" in self.cfg.env:
                                self.FRM_gen.metric_dict["socre"] = [round(episode_success, 2)]
                            else:
                                self.FRM_gen.metric_dict["socre"] = [round(true_episode_reward, 2)]
                            for metric in episode_reward_dict.keys():
                                self.FRM_gen.metric_dict[metric] = [round(episode_reward_dict[metric], 2)]
                        else:
                            if "metaworld" in self.cfg.env:
                                self.FRM_gen.metric_dict["socre"].append(
                                    round(episode_success, 2)
                                )
                            else:
                                self.FRM_gen.metric_dict["socre"].append(
                                    round(true_episode_reward, 2)
                                )

                            for metric in episode_reward_dict.keys():
                                if metric not in self.FRM_gen.metric_dict:
                                    print(f"[FRM] New metric detected: {metric}")
                                    self.FRM_gen.metric_dict[metric] = []

                                self.FRM_gen.metric_dict[metric].append(
                                    round(episode_reward_dict[metric], 2)
                                )

                        if self.step % self.cfg.FRM_align_step == 0:
                            # update FRM
                            eval_episode_freq = max(int(len(self.FRM_gen.metric_dict["socre"]) // 10), 1)
                            rf = self.FRM_gen.FRM_generate(episode_freq=eval_episode_freq, cur_frm=self.FRM)
                            if rf is not None:
                                # if improved, update the FRM
                                self.FRM = rf
                                # relabel
                                if "metaworld" in self.cfg.env:
                                    self.replay_buffer.relabel_with_rf(self.FRM, self.env.target_pos_frm, self.step)
                                elif "softgym_PassWater" in self.cfg.env:
                                    self.replay_buffer.relabel_with_rf(self.FRM, 1.2, self.step)
                                elif "softgym" in self.cfg.env:
                                    self.replay_buffer.relabel_with_rf(self.FRM, None, self.step)
                                else:
                                    raise ValueError
                                print("Relabel with new FRM successful!")
                                self.relabel_with_PRM()
                            else:
                                print("Using the last FRM! ")

                    self.logger.log('train_time/query_label', 0, self.step)
                    self.logger.log('train_time/RRM_learning', 0, self.step)
                    self.logger.log('train_time/relabel', 0, self.step)

                    self.logger.log('train/RRM_learning_acc', RRM_learning_acc, self.step)
                    self.logger.log('train/vlm_acc', vlm_acc, self.step)
                    if "Cloth" in self.cfg.env:
                        self.logger.log('train/true_reward', episode_end_reward, self.step)
                    
                # ====================================================
                # Evaluate agent periodically
                # ====================================================

                if "Cloth" in self.cfg.env:

                    # ------------------------------------------------
                    # ClothFoldDiagonal:
                    #
                    # Preserve the evaluation schedule used in the
                    # completed experiment:
                    #
                    # episode 250  -> step 750
                    # episode 500  -> step 1500
                    # ...
                    # episode 5000 -> step 15000
                    # ------------------------------------------------
                    should_evaluate = (
                        self.step > 0
                        and self.step % self.cfg.eval_frequency == 0
                    )

                else:

                    # ------------------------------------------------
                    # Other environments:
                    #
                    # Keep boundary-crossing evaluation so evaluation
                    # is not missed when an episode ends slightly
                    # after the requested boundary.
                    # ------------------------------------------------
                    should_evaluate = (
                        self.step > 0
                        and self.step // self.cfg.eval_frequency
                        > (self.step - episode_step)
                        // self.cfg.eval_frequency
                    )

                if should_evaluate:
                    self.evaluate(
                        episode
                    )

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/episode_reward_frm', episode_reward_frm, self.step)
                self.logger.log('train/episode_reward_rrm', episode_reward_rrm, self.step)
                self.logger.log('train/episode_reward_prm', episode_reward_prm, self.step)

                self.logger.log(
                    'train/PRM_loss',
                    self.latest_prm_loss,
                    self.step
                )

                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success, self.step)

                self.logger.dump(
                    self.step,
                    save=(self.step > self.cfg.num_seed_steps + 1),
                    ty='train'
                )

                # Store the completed episode return before checkpointing.
                avg_train_true_return.append(
                    true_episode_reward
                )

                # ====================================================
                # Automatic Checkpoint
                # ====================================================

                if (
                    self.step > 0
                    and self.step >= next_checkpoint_step
                ):
                    self.save_training_checkpoint(
                        episode=episode,
                        interact_count=interact_count,
                        avg_train_true_return=avg_train_true_return,
                        RRM_learning_acc=RRM_learning_acc,
                        vlm_acc=vlm_acc,
                    )

                    next_checkpoint_step = (
                        (
                            self.step
                            // self.checkpoint_interval
                        )
                        + 1
                    ) * self.checkpoint_interval

                # ====================================================
                # Reset Environment
                # ====================================================

                obs = self.env.reset()
                
                if "metaworld" in self.cfg.env:
                    obs = obs[0]
                # get goal pos
                if "metaworld" in self.cfg.env:
                    self.FRM_gen.target_pos = self.env.target_pos_frm
                elif "Water" in self.cfg.env:
                    self.FRM_gen.target_pos = 1.2

                done = False
                episode_reward, episode_reward_frm, episode_reward_rrm, episode_reward_prm = 0, 0, 0, 0
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                prev_progress = 0.0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                step_frm_dict = []

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update                
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_pre_steps):
                
                self.reward_alpha = self.cfg.reward_alpha
                self.replay_buffer.reward_alpha = self.cfg.reward_alpha
                print(f"set the rm_ alpha: {self.reward_alpha}")
                # update schedule
                if self.cfg.RRM_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.RRM_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.RRM.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.RRM.set_teacher_thres_skip(new_margin)
                self.RRM.set_teacher_thres_equal(new_margin)
                
                # first learn RRM
                RRM_learning_acc, vlm_acc = self.learn_RRM(first_flag=1)

                if self.use_PRM:
                    self.learn_PRM()

                # ====================================================
                # Relabel replay buffer
                # ====================================================
                print(
                    f"[RELABEL] Starting RRM relabel "
                    f"at step {self.step}..."
                )

                rrm_relabel_start = time.time()

                self.relabel_with_RRM()

                print(
                    f"[RELABEL] RRM relabel finished in "
                    f"{time.time() - rrm_relabel_start:.2f} seconds"
                )

                print(
                    f"[RELABEL] Starting PRM relabel "
                    f"at step {self.step}..."
                )

                prm_relabel_start = time.time()

                self.relabel_with_PRM()

                print(
                    f"[RELABEL] PRM relabel finished in "
                    f"{time.time() - prm_relabel_start:.2f} seconds"
                )

                # reset Q after reward relabeling
                print("Reset ACTOR AND CRITIC")
                
                self.agent.reset_critic()
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_pre_steps:
                # update rRRM
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.RRM_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.RRM_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.RRM.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.RRM.set_teacher_thres_skip(new_margin)
                        self.RRM.set_teacher_thres_equal(new_margin)
                        
                        # corner case: new total feed > max feed
                        if self.RRM.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.RRM.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        RRM_learning_acc, vlm_acc = self.learn_RRM()

                        if self.use_PRM:
                            self.learn_PRM()

                        relabel_timer = time.time()

                        print(
                            f"[RELABEL] Starting RRM relabel "
                            f"at step {self.step}..."
                        )

                        rrm_relabel_start = time.time()

                        self.relabel_with_RRM()

                        print(
                            f"[RELABEL] RRM relabel finished in "
                            f"{time.time() - rrm_relabel_start:.2f} seconds"
                        )

                        print(
                            f"[RELABEL] Starting PRM relabel "
                            f"at step {self.step}..."
                        )

                        prm_relabel_start = time.time()

                        self.relabel_with_PRM()

                        print(
                            f"[RELABEL] PRM relabel finished in "
                            f"{time.time() - prm_relabel_start:.2f} seconds"
                        )

                        self.logger.log(
                            'train_time/relabel',
                            time.time() - relabel_timer,
                            self.step
                        )

                        print(
                            f"[RELABEL] Total relabel time: "
                            f"{time.time() - relabel_timer:.2f} seconds"
                        )

                        interact_count = 0

            if not self.step < self.cfg.num_seed_steps:    
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)

            try: # for handle stupid gym wrapper change 
                next_obs, reward, done, extra = self.env.step(action)
            except:
                next_obs, reward, terminated, truncated, extra = self.env.step(action)
                done = terminated or truncated

            # Get RGB images for each step
            if "metaworld" in self.cfg.env:
                rgb_image = self.env.render()
                rgb_image = rgb_image[::-1, :, :]
                if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                    rgb_image = rgb_image[100:400, 100:400, :]
            elif 'softgym' in self.cfg.env:
                rgb_image = self.env.render(mode='rgb_array', hide_picker=True)
            else:
                rgb_image = self.env.render(mode='rgb_array')

            if 'Water' not in self.cfg.env and 'Rope' not in self.cfg.env:
                rgb_image = cv2.resize(rgb_image, (self.image_height, self.image_width)) # NOTE: resize image here

            if self.step < (self.cfg.num_seed_steps + self.cfg.num_pre_steps):
                    # not use RRM when first RRM learning
                    reward_rrm = 0
            else:
                image = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0
                image = image[:, ::self.resize_factor, ::self.resize_factor] # subsample
                image = image.reshape(1, 3, image.shape[1], image.shape[2])
                self.RRM.eval()
                reward_rrm = self.RRM.r_hat(image)
                self.RRM.train()

            # compute PRM reward
            reward_prm = 0.0
            if self.use_PRM and self.prm_ready:
                prm_image = rgb_image[
                    ::self.resize_factor,
                    ::self.resize_factor,
                    :
                ]
                progress = self.PRM.predict(prm_image)

                # Avoid one artificial jump immediately after PRM update.
                if self.prm_just_updated:
                    prev_progress = progress
                    self.prm_just_updated = False
                else:
                    reward_prm = (
                        self.PRM_gamma * progress
                        - prev_progress
                    )
                    reward_prm = float(np.clip(
                        reward_prm,
                        -self.PRM_clip,
                        self.PRM_clip
                    ))
                    prev_progress = progress

            # compute CoRe reward
            if "metaworld" in self.cfg.env:
                reward_frm, step_frm_info = self.FRM(next_obs, action, self.env.target_pos_frm)
            elif "softgym_PassWater" in self.cfg.env:
                reward_frm, step_frm_info = self.FRM(next_obs, action, 1.2)
            elif "softgym" in self.cfg.env:
                reward_frm, step_frm_info = self.FRM(next_obs, action, None)
            else:
                raise ValueError
            reward_frm = np.clip(reward_frm, -1, 1)
            step_frm_dict.append(step_frm_info)
            reward_frm = reward_frm

            # allow infinite bootstrap
            done = float(done)
            if 'softgym' not in self.cfg.env:
                done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            else:
                done_no_max = done

            # reward fusion
            reward_frm = (1.0 - self.reward_alpha) * reward_frm
            reward_rrm = self.reward_alpha * reward_rrm

            reward_prm = self.PRM_beta * reward_prm
            reward_hat = reward_frm + reward_rrm + reward_prm

            episode_reward += reward_hat
            episode_reward_frm += reward_frm
            episode_reward_rrm += reward_rrm
            episode_reward_prm += reward_prm
            
            if "Cloth" in self.cfg.env:
                true_episode_reward = reward
            else:
                true_episode_reward += reward

            if "Cloth" in self.cfg.env and episode_step == 2:
                episode_end_reward = reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data
            self.replay_buffer.add(obs, action, reward_hat, reward_frm, reward_rrm, 
                next_obs, done, done_no_max, image=rgb_image[::self.resize_factor, ::self.resize_factor, :])

            if len(self.RRM.all_cached_labels) < self.cfg.max_feedback and self.total_feedback < self.cfg.max_feedback:
                self.RRM.add_data(obs, action, reward, done, img=rgb_image)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

            if self.empty_pre_dataset and not self.RRM.preference_data_emptyed and self.total_feedback >= self.cfg.max_feedback:
                self.RRM.empty_preference_data()
            
        # ============================================================
        # Final Training Checkpoint
        # ============================================================

        self.save_training_checkpoint(
            episode=episode,
            interact_count=interact_count,
            avg_train_true_return=avg_train_true_return,
            RRM_learning_acc=RRM_learning_acc,
            vlm_acc=vlm_acc,
        )

        # ============================================================
        # Existing Final Model Saving
        # ============================================================

        self.agent.save(
            model_save_dir,
            self.step
        )

        self.RRM.save(
            model_save_dir,
            self.step
        )

        if self.use_PRM:
            self.PRM.save(
                model_save_dir,
                self.step
            )

        # ============================================================
        # Generate Final GIF
        # ============================================================

        self.save_final_demo_gif(
            num_episodes=3,
            fps=2
        )            
            
@hydra.main(config_path='config/train_CoRe.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
