from ttt.reward import fits_board, resource_pct, reward, safe


def test_safe_handles_none_and_garbage():
    assert safe(None, default=5.0) == 5.0
    assert safe("abc", default=1.0) == 1.0
    assert safe(float("nan"), default=2.0) == 2.0
    assert safe(3) == 3.0


def test_compile_failure_is_worst():
    assert reward({"compile_success": False}) == -1000.0


def test_accuracy_drift_is_heavily_penalized():
    r = reward({"compile_success": True, "max_error": 0.5})
    assert r < -500.0


def test_good_config_beats_resource_heavy_config():
    light = {
        "compile_success": True,
        "max_error": 0.02,
        "latency_cycles": 100,
        "dsp": 20,
        "lut": 3000,
        "bram": 4,
    }
    heavy = {
        "compile_success": True,
        "max_error": 0.02,
        "latency_cycles": 100,
        "dsp": 180,
        "lut": 45000,
        "bram": 100,
    }
    assert reward(light) > reward(heavy)


def test_lower_latency_is_rewarded():
    base = {"compile_success": True, "max_error": 0.01, "dsp": 10, "lut": 1000, "bram": 2}
    fast = reward({**base, "latency_cycles": 100})
    slow = reward({**base, "latency_cycles": 1000})
    assert fast > slow


def test_over_budget_penalty_applied():
    on_budget = {"compile_success": True, "max_error": 0.01, "latency_cycles": 100, "dsp": 100, "lut": 1000, "bram": 2}
    over_budget = {**on_budget, "dsp": 210}  # > 0.8 * 220
    assert reward(over_budget) < reward(on_budget) - 200


def test_missing_resources_do_not_crash():
    r = reward({"compile_success": True, "max_error": 0.0})
    assert isinstance(r, float)


def test_fits_board():
    assert fits_board({"dsp": 50, "lut": 1000}) is True
    assert fits_board({"dsp": 5000}) is False
    assert fits_board({"dsp": None}) is True  # unknown => assume fits


def test_resource_pct():
    pct = resource_pct({"dsp": 110})
    assert abs(pct["dsp_pct"] - 50.0) < 1e-6
