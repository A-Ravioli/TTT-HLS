from ttt.config_space import sample_random_configs
from ttt.online_policy import OnlineTTTPolicy
from ttt.reward import reward


def _fake_reward(cfg):
    # Synthetic objective: prefer Latency strategy + low reuse (proxy for low latency).
    r = {
        "compile_success": True,
        "max_error": 0.01,
        "latency_cycles": 50 * (cfg.reuse_dense_1 + cfg.reuse_dense_2),
        "dsp": 10,
        "lut": 1000,
        "bram": 2,
    }
    return reward(r)


def test_propose_before_fit_returns_configs():
    policy = OnlineTTTPolicy(random_state=1)
    proposals = policy.propose(n=3)
    assert len(proposals) == 3
    assert not policy.has_fit


def test_propose_excludes_tried():
    policy = OnlineTTTPolicy(random_state=1)
    tried = {c.short_name() for c in sample_random_configs(3)}
    proposals = policy.propose(n=3, exclude=tried)
    assert all(c.short_name() not in tried for c in proposals)


def test_policy_learns_to_prefer_high_reward():
    policy = OnlineTTTPolicy(random_state=0)
    configs = sample_random_configs(40)
    rewards = [_fake_reward(c) for c in configs]
    assert policy.fit(configs, rewards)
    assert policy.has_fit

    proposals = policy.propose(n=5, n_candidates=300)
    proposed_rewards = [_fake_reward(c) for c in proposals]
    # The fitted policy should propose better-than-average configs.
    assert sum(proposed_rewards) / len(proposed_rewards) > sum(rewards) / len(rewards)
