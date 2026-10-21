import os
import shutil
import numpy as np
import torch
import utils

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, image_size=300, reward_alpha=0.5):
        self.capacity = capacity
        self.device = device
        self.image_size = image_size
        # for time decrease
        self.reward_alpha = reward_alpha

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)

        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)

        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards_frm = np.empty((capacity, 1), np.float32)
        self.rewards_rrm = np.empty((capacity, 1), np.float32)

        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.images_path = None

        if image_size:

            replay_memmap_dir = os.environ.get(
                "REPLAY_MEMMAP_DIR",
                os.environ.get(
                    "RRM_MEMMAP_DIR",
                    os.environ.get(
                        "SCRATCH_DIR",
                        os.environ.get(
                            "TMPDIR",
                            "/tmp"
                        )
                    )
                )
            )

            os.makedirs(
                replay_memmap_dir,
                exist_ok=True
            )

            self.images_path = os.path.join(
                replay_memmap_dir,
                f"core_replay_images_{os.getpid()}.dat"
            )

            self.images = np.memmap(
                self.images_path,
                dtype=np.uint8,
                mode="w+",
                shape=(
                    capacity,
                    image_size,
                    image_size,
                    3
                )
            )

            print(
                f"[ReplayBuffer] Using disk-backed image buffer: "
                f"{self.images_path}"
            )

        else:
            self.images = None

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, reward_frm, reward_rrm, 
            next_obs, done, done_no_max, image=None):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)

        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.rewards_frm[self.idx], reward_frm)
        np.copyto(self.rewards_rrm[self.idx], reward_rrm)

        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        if image is not None and self.images is not None:
            np.copyto(self.images[self.idx], image)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        if isinstance(self.images, np.memmap) and self.idx % 1000 == 0:
            self.images.flush()        

    def relabel_with_predictor(self, predictor, step):

        batch_size = 32
        relabel_max_idx = self.idx if not self.full else self.capacity
        total_iter = int(np.ceil(relabel_max_idx / batch_size))

        for index in range(total_iter):

            start_index = index * batch_size
            last_index = (index + 1) * batch_size

            if last_index > relabel_max_idx:
                last_index = relabel_max_idx

            # Read only a small portion of the disk-backed image buffer.
            inputs = self.images[start_index:last_index]

            inputs = np.transpose(
                inputs,
                (0, 3, 1, 2)
            )

            # uint8 -> float32 only for this small batch.
            inputs = inputs.astype(np.float32) / 255.0

            pred_reward = predictor.r_hat_batch(inputs)

            self.rewards_rrm[
                start_index:last_index
            ] = pred_reward

            self.rewards[
                start_index:last_index
            ] = (
                (1.0 - self.reward_alpha)
                * self.rewards_frm[start_index:last_index]
                +
                self.reward_alpha
                * self.rewards_rrm[start_index:last_index]
            )

            # Release temporary batch memory immediately.
            del inputs
            del pred_reward

        torch.cuda.empty_cache()
    
    def relabel_with_rf(self, frm, target_pos, step):
        relabel_max_idx = self.idx if not self.full else self.capacity
        for idx in range(relabel_max_idx):
            self.rewards_frm[idx], _ = frm(self.obses[idx], self.actions[idx], target_pos)
            self.rewards[idx] = (1.0 - self.reward_alpha) * self.rewards_frm[idx] + self.reward_alpha * self.rewards_rrm[idx]
    
    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
    
    def sample_state_ent(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)
        
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_obs = torch.as_tensor(full_obs, device=self.device)
        
        return obses, full_obs, actions, rewards, next_obses, not_dones, not_dones_no_max

    def save(self, save_dir):
        """
        Save replay buffer state for checkpoint/resume.

        Numeric replay data is stored as .npy files.

        The large image buffer is copied directly from disk to disk.
        It is NOT loaded into RAM.
        """

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        size = (
            self.capacity
            if self.full
            else self.idx
        )

        print(
            f"[ReplayBuffer] Saving {size} transitions..."
        )

        # ============================================================
        # Save Replay Buffer Metadata
        # ============================================================

        metadata_path = os.path.join(
            save_dir,
            "metadata.npz"
        )

        np.savez(
            metadata_path,
            idx=self.idx,
            full=self.full,
            capacity=self.capacity,
            size=size,
            reward_alpha=self.reward_alpha,
            image_size=(
                self.image_size
                if self.image_size is not None
                else -1
            ),
            has_images=(
                self.images is not None
            ),
        )

        # ============================================================
        # Save Numeric Replay Data
        # ============================================================

        np.save(
            os.path.join(
                save_dir,
                "obses.npy"
            ),
            self.obses[:size]
        )

        np.save(
            os.path.join(
                save_dir,
                "next_obses.npy"
            ),
            self.next_obses[:size]
        )

        np.save(
            os.path.join(
                save_dir,
                "actions.npy"
            ),
            self.actions[:size]
        )

        np.save(
            os.path.join(
                save_dir,
                "rewards.npy"
            ),
            self.rewards[:size]
        )

        np.save(
            os.path.join(
                save_dir,
                "rewards_frm.npy"
            ),
            self.rewards_frm[:size]
        )

        np.save(
            os.path.join(
                save_dir,
                "rewards_rrm.npy"
            ),
            self.rewards_rrm[:size]
        )

        np.save(
            os.path.join(
                save_dir,
                "not_dones.npy"
            ),
            self.not_dones[:size]
        )

        np.save(
            os.path.join(
                save_dir,
                "not_dones_no_max.npy"
            ),
            self.not_dones_no_max[:size]
        )

        # ============================================================
        # Save Disk-Backed Image Buffer
        # ============================================================

        if (
            self.images is not None
            and self.images_path is not None
        ):
            if isinstance(
                self.images,
                np.memmap
            ):
                self.images.flush()

            image_checkpoint_path = os.path.join(
                save_dir,
                "images.dat"
            )

            print(
                "[ReplayBuffer] Saving image memmap "
                "disk-to-disk..."
            )

            shutil.copyfile(
                self.images_path,
                image_checkpoint_path
            )

            print(
                f"[ReplayBuffer] Image snapshot saved: "
                f"{image_checkpoint_path}"
            )

        print(
            f"[ReplayBuffer] Checkpoint saved "
            f"with {size} transitions."
        )


    def load(self, save_dir):
        """
        Restore replay buffer from checkpoint.

        The checkpoint image file is first copied into the
        current live memmap file. This keeps the checkpoint
        itself immutable while training continues.
        """

        print(
            f"[ReplayBuffer] Loading checkpoint from: "
            f"{save_dir}"
        )

        # ============================================================
        # Load Metadata
        # ============================================================

        metadata_path = os.path.join(
            save_dir,
            "metadata.npz"
        )

        metadata = np.load(
            metadata_path
        )

        saved_capacity = int(
            metadata["capacity"]
        )

        saved_size = int(
            metadata["size"]
        )

        saved_idx = int(
            metadata["idx"]
        )

        saved_full = bool(
            metadata["full"]
        )

        saved_reward_alpha = float(
            metadata["reward_alpha"]
        )

        saved_image_size = int(
            metadata["image_size"]
        )

        has_images = bool(
            metadata["has_images"]
        )

        metadata.close()

        # ============================================================
        # Validate Configuration
        # ============================================================

        if saved_capacity != self.capacity:
            raise RuntimeError(
                f"Replay buffer capacity mismatch. "
                f"Checkpoint={saved_capacity}, "
                f"Current={self.capacity}"
            )

        if (
            has_images
            and saved_image_size != self.image_size
        ):
            raise RuntimeError(
                f"Replay image size mismatch. "
                f"Checkpoint={saved_image_size}, "
                f"Current={self.image_size}"
            )

        # ============================================================
        # Load Numeric Replay Data
        # ============================================================

        self.obses[:saved_size] = np.load(
            os.path.join(
                save_dir,
                "obses.npy"
            ),
            mmap_mode="r"
        )

        self.next_obses[:saved_size] = np.load(
            os.path.join(
                save_dir,
                "next_obses.npy"
            ),
            mmap_mode="r"
        )

        self.actions[:saved_size] = np.load(
            os.path.join(
                save_dir,
                "actions.npy"
            ),
            mmap_mode="r"
        )

        self.rewards[:saved_size] = np.load(
            os.path.join(
                save_dir,
                "rewards.npy"
            ),
            mmap_mode="r"
        )

        self.rewards_frm[:saved_size] = np.load(
            os.path.join(
                save_dir,
                "rewards_frm.npy"
            ),
            mmap_mode="r"
        )

        self.rewards_rrm[:saved_size] = np.load(
            os.path.join(
                save_dir,
                "rewards_rrm.npy"
            ),
            mmap_mode="r"
        )

        self.not_dones[:saved_size] = np.load(
            os.path.join(
                save_dir,
                "not_dones.npy"
            ),
            mmap_mode="r"
        )

        self.not_dones_no_max[:saved_size] = np.load(
            os.path.join(
                save_dir,
                "not_dones_no_max.npy"
            ),
            mmap_mode="r"
        )

        # ============================================================
        # Restore Replay Buffer State
        # ============================================================

        self.idx = saved_idx
        self.full = saved_full
        self.reward_alpha = saved_reward_alpha

        # ============================================================
        # Restore Image Memmap
        # ============================================================

        if has_images:
            image_checkpoint_path = os.path.join(
                save_dir,
                "images.dat"
            )

            if not os.path.exists(
                image_checkpoint_path
            ):
                raise RuntimeError(
                    f"Replay image checkpoint missing: "
                    f"{image_checkpoint_path}"
                )

            if self.images is None:
                raise RuntimeError(
                    "Current ReplayBuffer does not have "
                    "an image buffer."
                )

            print(
                "[ReplayBuffer] Restoring image memmap "
                "from checkpoint..."
            )

            # Flush and close the empty memmap created
            # when ReplayBuffer was initialized.
            if isinstance(
                self.images,
                np.memmap
            ):
                self.images.flush()

                try:
                    self.images._mmap.close()
                except Exception:
                    pass

            # Copy checkpoint snapshot into the current
            # live memmap file.
            shutil.copyfile(
                image_checkpoint_path,
                self.images_path
            )

            # Reopen the restored live image buffer.
            self.images = np.memmap(
                self.images_path,
                dtype=np.uint8,
                mode="r+",
                shape=(
                    self.capacity,
                    self.image_size,
                    self.image_size,
                    3
                )
            )

        print(
            f"[ReplayBuffer] Restored "
            f"{saved_size} transitions."
        )

        print(
            f"[ReplayBuffer] idx={self.idx}, "
            f"full={self.full}, "
            f"reward_alpha={self.reward_alpha}"
        )        
