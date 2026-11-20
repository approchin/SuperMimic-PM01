# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Share reset difficulty between original and mirrored variants of each source-time bin."""
import json
from pathlib import Path

import torch


class SourceBinSampler:
    groups = ("ordinary", "run_or_transition", "jump")

    def __init__(self, path, motion, device):
        spec = json.loads(Path(path).read_text())
        clip_names = list(motion.names)
        # Map by stable clip name: the HDF5 Dataset may sort names independently of JSON insertion order.
        name_to_id = {name:i for i,name in enumerate(clip_names)}
        if len(name_to_id) != len(clip_names):
            raise ValueError("Duplicate live motion names")
        length = motion.clip_len.tolist()
        families = {}
        offset, group_index, variant_count = [], [], []
        self.probabilities = []
        source_groups = []
        for group_id, group_name in enumerate(self.groups):
            group = spec["groups"][group_name]
            self.probabilities.append(float(group["probability"]))
            for interval in group["source_intervals"]:
                names = interval["clip_ids"]
                bins = int(interval["one_second_bin_count"])
                first = len(source_groups)
                source_groups.extend([group_id]*bins)
                for name in names:
                    i = name_to_id[name]
                    if name in families or length[i] != interval["length_frames"] or bins != (length[i]-2)//50+1:
                        raise ValueError("Invalid repeated source identity or source bin length: "+name)
                    families[name] = (first, group_id, len(names))
        if set(families) != set(clip_names) or self.probabilities != [.90,.05,.05]:
            raise ValueError("011 sampler requires complete clip coverage and the approved group mixture")
        for name in clip_names:
            a,b,c = families[name]
            offset.append(a); group_index.append(b); variant_count.append(c)
        tensor = lambda x: torch.tensor(x, device=device, dtype=torch.long)
        self.clip_group = tensor(group_index)
        self.bin_source = tensor(offset)[motion.bin_clip] + (motion.bin_start-motion.clip_start[motion.bin_clip])//50
        self.variant_count = tensor(variant_count)[motion.bin_clip]
        self.source_group = tensor(source_groups)
        self.group_indices = [(self.source_group==g).nonzero().flatten() for g in range(len(self.groups))]
        self.representative = torch.empty(len(source_groups), dtype=torch.long, device=device)
        self.representative.scatter_(0, self.bin_source, torch.arange(motion.num_bins,device=device))
        self.reset_counts = torch.zeros(3, dtype=torch.long, device=device)
        self.transition_counts = torch.zeros_like(self.reset_counts)

    def initialize(self, env):
        env.bin_G.zero_()
        avg = torch.empty_like(self.source_group, dtype=env.bin_w_avg.dtype)
        for ids in self.group_indices:
            avg[ids] = 1./len(ids)
        env.bin_w_avg.copy_(avg[self.bin_source])

    def sample(self, env, count):
        returns = env.bin_G[self.representative]
        historic = env.bin_w_avg[self.representative]
        probability = torch.empty_like(returns)
        for ids, mass in zip(self.group_indices, self.probabilities):
            now = torch.softmax(-25*returns[ids]/env.G_max, 0)
            avg = .999*historic[ids]+.001*now
            historic[ids] = avg
            probability[ids] = mass*(.5/len(ids)+.4*now+.1*avg/avg.sum())
        env.bin_w_avg.copy_(historic[self.bin_source])
        weights = probability[self.bin_source] / self.variant_count
        return torch.multinomial(weights, count, replacement=True)

    def update(self, env, bins, values):
        family = self.bin_source[bins]
        sums = torch.zeros_like(self.source_group,dtype=values.dtype)
        counts = torch.zeros_like(sums)
        sums.scatter_add_(0,family,values)
        counts.scatter_add_(0,family,torch.ones_like(values))
        old = env.bin_G[self.representative]
        new = torch.where(counts>0,.8*old+.2*sums/counts.clamp_min(1),old)
        env.bin_G.copy_(new[self.bin_source])

    def count(self, counter, clips):
        groups = self.clip_group[clips]
        counter.scatter_add_(0, groups, torch.ones_like(groups))
