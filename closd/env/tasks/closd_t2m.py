# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from closd.env.tasks import closd_task
from isaacgym.torch_utils import *
from closd.utils.closd_util import STATES

class CLoSDT2M(closd_task.CLoSDTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.init_state = STATES.TEXT2MOTION
        self.custom_prompt = self.cfg['env']['dip'].get('custom_prompt', '')  # if non-empty, overrides dataset-sampled prompts
        # optional ordered queue of prompts (e.g. "squat" then "clap"): each entry plays for its
        # own frame duration, then update_state_machine() swaps in the next one mid-episode. The
        # pose buffer already tracks the real simulated pose, so the next prompt is naturally
        # planned onward from wherever the previous one left the character.
        self.prompt_sequence = [p for p in self.cfg['env']['dip'].get('custom_prompt_sequence', '').split('|') if p != '']
        if self.prompt_sequence:
            _frames_str = self.cfg['env']['dip'].get('custom_prompt_sequence_frames', '')
            if _frames_str:
                self.prompt_durations = [int(f) for f in _frames_str.split('|')]
                assert len(self.prompt_durations) == len(self.prompt_sequence), \
                    'custom_prompt_sequence_frames must have the same number of |-separated entries as custom_prompt_sequence'
            else:
                self.prompt_durations = [self.cfg['env']['dip'].get('default_segment_frames', 90)] * len(self.prompt_sequence)
            self.seq_idx = torch.zeros([self.num_envs], dtype=torch.int64, device=self.device)
            self.seq_switch_frame = torch.zeros([self.num_envs], dtype=torch.int64, device=self.device)
        self.hml_data_buf_size = max(self.fake_mdm_args.context_len, self.planning_horizon_20fps)
        # hml_prefix_from_data seeds the first planning steps with a real dataset motion. It is
        # only meaningful when prompts come from the dataset; downstream code in closd.py gates
        # on hasattr(self, 'hml_prefix_from_data'), so for a custom prompt we deliberately leave
        # it undefined and behave like closd_multitask (pose buffer seeded from the real pose).
        if self.custom_prompt == '' and not self.prompt_sequence:
            self.hml_prefix_from_data = torch.zeros([self.num_envs, 263, 1, self.hml_data_buf_size], dtype=torch.float32, device=self.device)
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        if self.prompt_sequence:
            self.seq_idx[env_ids] = 0
            self.seq_switch_frame[env_ids] = 0
        return

    def update_state_machine(self):
        if not self.prompt_sequence:
            return
        durations = torch.tensor(self.prompt_durations, device=self.device, dtype=self.progress_buf.dtype)
        elapsed = self.progress_buf - self.seq_switch_frame
        not_last = self.seq_idx < (len(self.prompt_sequence) - 1)
        advance = (elapsed >= durations[self.seq_idx]) & not_last
        if torch.any(advance):
            self.seq_idx[advance] += 1
            self.seq_switch_frame[advance] = self.progress_buf[advance]
            self.update_mdm_conditions(advance.nonzero(as_tuple=False).squeeze(-1))
        return

    def update_mdm_conditions(self, env_ids):
        super().update_mdm_conditions(env_ids)

        if self.prompt_sequence:
            for i in env_ids:
                self.hml_prompts[int(i)] = self.prompt_sequence[int(self.seq_idx[int(i)])]
            if self.cfg['env']['dip']['debug_hml']:
                print(f'in update_mdm_conditions: seq prompt for env_ids={env_ids[:10]}, idx={self.seq_idx[env_ids[:10]].cpu().numpy()}')
            return

        if self.custom_prompt != '':
            # user-defined prompt: apply the same text to every env and skip the dataset.
            # The dataset lengths/tokens/db_keys are only evaluation bookkeeping, and the real
            # motion prefix (hml_prefix_from_data) is not needed because the pose buffer is
            # already seeded from the character's own pose in _reset_pose_buffer.
            for i in env_ids:
                self.hml_prompts[int(i)] = self.custom_prompt
            if self.cfg['env']['dip']['debug_hml']:
                print(f'in update_mdm_conditions: using custom_prompt for env_ids={env_ids[:10].cpu().numpy()}, prompt={self.custom_prompt!r}')
            return

        # updates prompts and lengths
        try:
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        except StopIteration:
            del self.mdm_data_iter
            self.mdm_data_iter = iter(self.mdm_data) # re-initialize
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        for i in env_ids:
            self.hml_prompts[int(i)] = model_kwargs['y']['text'][int(i)]
            self.hml_lengths[int(i)] = model_kwargs['y']['lengths'][int(i)]
            self.hml_tokens[int(i)] = model_kwargs['y']['tokens'][int(i)]
            self.db_keys[int(i)] = model_kwargs['y']['db_key'][int(i)]
        self.hml_prefix_from_data[env_ids] = gt_motion[..., :self.hml_data_buf_size].to(self.device)[env_ids]  # will be used by the first MDM iteration
        if self.cfg['env']['dip']['debug_hml']:
            print(f'in update_mdm_conditions: 1st 10 env_ids={env_ids[:10].cpu().numpy()}, prompts={self.hml_prompts[:2]}')
        return
    
    def get_cur_done(self):
        # Done signal is not in use for this task
        return torch.zeros([self.num_envs], device=self.device, dtype=bool)
    

