from hydrion.service.app import start_run, RunRequest

if __name__ == "__main__":
    req = RunRequest(policy_type="ppo", seed=123, config_name="default.yaml", max_steps=5, noise_enabled=False)
    res = start_run(req)
    print("started", res)
