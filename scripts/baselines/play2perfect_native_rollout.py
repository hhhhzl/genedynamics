"""Full-denominator native in-hand checkpoint rollout, prepared before EULA launch.

This wrapper never sets EULA acceptance. The caller must have explicitly accepted
and supplied its environment variable. Official model/environment algorithms stay
in the locked source checkout. Do not invoke until runtime and consent are ready.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import re

ROOT = Path(__file__).resolve().parents[2]


def verify_vulkan_gpu_log(path, expected_uuid, physical_index=1, expected_pci_bus_id=None):
    """Verify the active Vulkan row and its UUID, independently of CUDA ordinals."""
    expected = expected_uuid.removeprefix("GPU-").replace("-", "").lower()
    lines = path.read_text(errors="replace").splitlines()
    for start, line in enumerate(lines):
        cells = line.split("|")
        if len(cells) < 4 or cells[1].strip() != str(physical_index) or "yes" not in cells[3].lower():
            continue
        block = [line]
        for following in lines[start + 1:]:
            fields = following.split("|")
            if len(fields) > 1 and fields[1].strip().isdigit(): break
            block.append(following)
            if len(block) >= 5: break
        full_match = any(re.sub(r"[^0-9a-f]", "", cell.lower()) == expected
                         for row in block for cell in row.split("|"))
        prefix_match = any(cell.strip().endswith("..") and len(re.sub(r"[^0-9a-f]", "", cell.lower())) >= 8
                           and expected.startswith(re.sub(r"[^0-9a-f]", "", cell.lower()))
                           for row in block for cell in row.split("|"))
        bus_match = (expected_pci_bus_id is not None and any(
            len(row.split("|")) > 6 and row.split("|")[6].strip().lower() == expected_pci_bus_id.split(":")[-2].lower()
            for row in block))
        if full_match or (prefix_match and bus_match):
            return {"verified": True, "physical_gpu": physical_index, "uuid": expected_uuid,
                    "pci_bus_id": expected_pci_bus_id, "abbreviated_uuid_matched_via_NVML_PCI": not full_match,
                    "log": str(path), "gpu_foundation_rows": block}
    raise RuntimeError("Active Vulkan GPU table/UUID does not verify the selected physical GPU1")


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def native_fixed_goal_outcome(goals, achieved, *, retract_enabled, terminal_retract_success):
    """Retain the native fixed-goal insertion AND release/retract success gate."""
    insertion = goals > 0 and achieved >= goals
    if retract_enabled and terminal_retract_success is None:
        raise RuntimeError("Native retract-enabled task requires its pre-reset terminal retract snapshot")
    retract = bool(terminal_retract_success) if retract_enabled else None
    return {"goal_count": goals, "goals_achieved": achieved, "insertion_success": insertion,
            "retract_enabled": retract_enabled, "terminal_retract_success": retract,
            "success": insertion and (retract if retract_enabled else True)}


def full_denominator_summary(rows, requested):
    if len(rows) != requested or requested <= 0:
        raise ValueError("Every requested native trial must retain exactly one outcome row")
    return {"requested_trials": requested, "completed_trials": sum(row["ended"] for row in rows),
            "unfinished_trials": sum(not row["ended"] for row in rows),
            "insertion_successes": sum(bool(row.get("insertion_success", False)) for row in rows),
            "successes": sum(bool(row["success"]) for row in rows),
            "full_denominator_success_rate": sum(bool(row["success"]) for row in rows) / requested}


def first_episode_rows(requested):
    return [dict(slot=i, ended=False, success=False, insertion_success=False,
                 first_episode_steps=0, censored=True, status="not_started") for i in range(requested)]


def coefficient_assignment(coefficients, trained_ids):
    """Audit the source network's exact equality followed by first-index argmax."""
    result = []
    for slot, value in enumerate(coefficients):
        matches = [i for i, trained in enumerate(trained_ids) if value == trained]
        index = matches[0] if matches else 0
        result.append(dict(slot=slot, coefficient=value, resolved_parameter_id=index,
                           resolved_trained_coefficient=trained_ids[index], exact_match=bool(matches)))
    return result


def partial_report(state, plan):
    """A failure retains all requested slots, including those never started."""
    rows = state["first_episode_rows"]
    return {**plan, **full_denominator_summary(rows, state["requested_trials"]), "trials": rows,
            "attempt_status": state["status"], "physical_qualification_complete": False,
            "physics": state["physics"], "actual_control_calls": state["control_calls"],
            "in_flight_physics_step": state["in_flight_physics_step"],
            "in_flight_control_step": state["in_flight_control_step"],
            "scope": "Partial observed outcomes; censored or not-started slots remain in requested denominator."}


def launch_bounded(args, local, plan):
    """Own the entire worker lifetime, including Kit startup and teardown."""
    import psutil
    receipt = args.output / "native_attempt.json"
    if receipt.exists(): raise RuntimeError("A started native attempt cannot be reused; choose a new seed/output")
    started = time.monotonic()
    state = {"status": "launching", "controller_pid": os.getpid(), "requested_trials": args.num_envs,
             "gpu": args.gpu, "device": "cuda:0", "physics": {"setup_physics_step_calls": 0, "rollout_physics_step_calls": 0},
             "control_calls": 0, "in_flight_physics_step": False, "in_flight_control_step": False,
             "first_episode_rows": first_episode_rows(args.num_envs)}
    atomic_json(receipt, state)
    command = [sys.executable, "-m", "scripts.baselines.play2perfect_native_rollout", *sys.argv[1:], "--worker"]
    local["P2P_NATIVE_CONTROLLER_PID"] = str(os.getpid())
    profile = {"peak_owned_gpu_mib": 0, "peak_owned_rss_bytes": 0,
               "max_seconds": args.max_seconds, "max_gpu_mib": args.max_gpu_mib, "max_rss_gib": args.max_rss_gib,
               "setup_time_included": True, "command": command, "plan": plan}
    failure, child = None, None
    with (args.output / "native_worker.log").open("w") as log:
        try:
            child = subprocess.Popen(command, cwd=ROOT, env=local, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            gpu_inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=5)
            allowed_uuid = next(line.split(",")[1].strip() for line in gpu_inventory.splitlines() if line.split(",")[0].strip() == "1")
            while child.poll() is None:
                try:
                    processes = [psutil.Process(child.pid)]
                    processes.extend(processes[0].children(recursive=True))
                except psutil.NoSuchProcess:
                    if child.poll() is not None: break
                    continue
                pids, rss = set(), 0
                for process in processes:
                    pids.add(process.pid)
                    try: rss += process.memory_info().rss
                    except psutil.NoSuchProcess: pass
                applications = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], text=True, timeout=5)
                used = 0
                for row in applications.splitlines():
                    pieces = [item.strip() for item in row.split(",")]
                    if len(pieces) != 3 or not pieces[0].isdigit() or int(pieces[0]) not in pids: continue
                    if pieces[1] != allowed_uuid:
                        profile["device_violation"] = {"pid": int(pieces[0]), "gpu_uuid": pieces[1], "used_memory_mib": pieces[2]}
                        raise RuntimeError("Worker allocated compute memory outside physical GPU1")
                    if pieces[2].isdigit(): used += int(pieces[2])
                elapsed = time.monotonic() - started
                profile.update(wall_seconds=elapsed, worker_pid=child.pid,
                    peak_owned_gpu_mib=max(profile["peak_owned_gpu_mib"], used),
                    peak_owned_rss_bytes=max(profile["peak_owned_rss_bytes"], rss))
                atomic_json(args.output / "native_launch.json", profile)
                if elapsed > args.max_seconds or used > args.max_gpu_mib or rss > args.max_rss_gib * 1024**3:
                    raise RuntimeError("Native worker exceeded setup-inclusive wall/GPU/RSS budget")
                time.sleep(1)
        except BaseException as error:
            failure = {"type": type(error).__name__, "message": str(error)}
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            current = json.loads(receipt.read_text())
            returncode = child.returncode if child is not None else None
            current.update(worker_returncode=returncode, wall_seconds=time.monotonic() - started)
            if failure is not None: current.update(status="terminated_by_watchdog", watchdog_exception=failure)
            elif returncode or current["status"] != "rollout_completed" or not (args.output / "native_closed_loop.json").is_file():
                current["status"] = "worker_failed"
                failure = current.get("exception", {"type": "IncompleteNativeQualification", "message": "Worker exit0 is insufficient without completed rollout evidence"})
            else: current["status"] = "complete"
            current["accounting_limit"] = "Confirmed completed native step calls retained; in-flight flags identify operations interrupted before return."
            atomic_json(receipt, current)
            if current["status"] != "complete":
                atomic_json(args.output / "native_partial_closed_loop.json", partial_report(current, plan))
            profile.update(returncode=returncode, wall_seconds=time.monotonic() - started, failure=failure)
            atomic_json(args.output / "native_launch.json", profile)
    if failure is not None or returncode: raise RuntimeError("Native attempt failed; retained native_attempt/native_launch/native_worker evidence")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2): h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=ROOT / ".runtime/play2perfect_native")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coefficient-mode", choices=("source-mixed", "exploitation"), default="exploitation")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--max-gpu-mib", type=int, default=10240)
    parser.add_argument("--max-rss-gib", type=float, default=20)
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.max_seconds, args.max_gpu_mib, args.max_rss_gib) <= 0 or args.gpu != 1:
        parser.error("bounded native qualification requires physical GPU1 and positive resource budgets")
    coefficients = ([0.] * args.num_envs if args.coefficient_mode == "exploitation" else
                    [50. - 50. * i / max(args.num_envs - 1, 1) for i in range(args.num_envs)])
    assignments = coefficient_assignment(coefficients, [50., 40., 30., 20., 10., 0.])
    if not all(row["exact_match"] for row in assignments):
        parser.error("source-mixed slot coefficients must exactly match trained IDs; use --num-envs 6 or explicit exploitation mode")
    args.output, args.runtime, args.checkpoint = args.output.resolve(), args.runtime.resolve(), args.checkpoint.resolve()
    if "latex" in args.output.parts: parser.error("native receipts belong outside the paper")
    from scripts.baselines.sources import LOCK, verify
    spec = next(item for item in json.loads(LOCK.read_text())["sources"] if item["method"] == "play2perfect")
    source_receipt = verify(spec)
    resources = json.loads((ROOT / "requirements/baselines/resources.lock.json").read_text())["resources"]
    asset = next(row for row in resources if row["id"] == "play2perfect_tight_insertion")
    if sha256(args.checkpoint) != asset["sha256"]: raise RuntimeError("Expected the locked published tight-insertion checkpoint")
    plan = {"schema": "play2perfect-native-full-denominator-plan-v1", "source": source_receipt,
        "checkpoint_sha256": asset["sha256"], "seed": args.seed, "num_envs": args.num_envs, "max_control_steps": args.steps,
        "AppLauncher": {"device": "cuda:0", "multi_gpu": False, "headless": True, "enable_cameras": False},
        "physical_gpu": args.gpu, "cuda_visibility": "only the physical GPU1 UUID; logical CUDA ordinal0",
        "env_device": "cuda:0", "actor_device": "cuda:0",
        "in_hand_model": "original IIWA7 + Sharpa22; independent native object contacts and original action/reset logic",
        "coefficient_mode": args.coefficient_mode, "coefficient_assignment": assignments,
        "coefficient_scope": "Exploitation explicitly selects trained coefficient zero in every slot. Source-mixed preserves original linspace only when every slot matches a trained ID.",
        "denominator": "every requested first episode; early resets and unfinished trials retained",
        "limits": {"wall_seconds_including_setup": args.max_seconds, "owned_gpu_mib": args.max_gpu_mib, "owned_rss_gib": args.max_rss_gib},
        "Kit_args": f"--portable-root {args.runtime / 'kit-portable'} --/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false --/app/settings/loadUserConfig=false",
        "EULA": "wrapper refuses to start without caller-provided acceptance; never sets automatic YES",
        "cache_scope": "owned runtime XDG/Kit/TMP/CUDA/GL; USD conversion uses TMPDIR"}
    if args.plan_only:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "native_rollout_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2))
        return
    acceptance = os.environ.get("OMNI_KIT_ACCEPT_EULA", "")
    if acceptance.upper() not in {"YES", "Y", "1"}:
        raise RuntimeError("Isaac Sim EULA has not been explicitly accepted by the caller; refusing Kit startup")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"], text=True, timeout=5)
    devices = [[cell.strip() for cell in line.split(",")] for line in inventory.splitlines()]
    selected = next(row for row in devices if row[0] == str(args.gpu))
    selected_uuid, selected_pci = selected[1:]
    if sum(row[2].split(":")[-2] == selected_pci.split(":")[-2] for row in devices) != 1:
        raise RuntimeError("Vulkan abbreviated UUID requires a unique NVML PCI bus mapping")
    if args.worker and os.environ.get("CUDA_VISIBLE_DEVICES") != selected_uuid:
        raise RuntimeError("Native worker must expose only the selected physical GPU1 UUID")
    plan["selected_gpu_uuid"] = selected_uuid
    plan["selected_gpu_pci_bus_id"] = selected_pci
    if (args.output / "native_closed_loop.json").exists():
        raise RuntimeError("A physical qualification receipt exists; use a new config-mirrored seed/output")
    from scripts.baselines.play2perfect_native_environment import owned_environment
    local = owned_environment(args.runtime)
    local["OMNI_KIT_ACCEPT_EULA"] = acceptance  # preserve explicit caller authorization, never invent it
    local["CUDA_VISIBLE_DEVICES"] = selected_uuid
    os.environ.update(local)
    sys.pycache_prefix = local["PYTHONPYCACHEPREFIX"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "native_rollout_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not args.worker:
        launch_bounded(args, local, plan)
        return
    controller = os.environ.get("P2P_NATIVE_CONTROLLER_PID")
    receipt = args.output / "native_attempt.json"
    if controller != str(os.getppid()) or not receipt.is_file():
        raise RuntimeError("Native worker must be started by its bounded controller")
    state = json.loads(receipt.read_text())
    state.update(worker_pid=os.getpid(), phase="worker initialization", status="running")
    def persist(): atomic_json(receipt, state)
    persist()
    try:
        _run_native_worker(args, plan, state, persist)
    except BaseException as error:
        state.update(status="failed", exception={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        persist()


def _run_native_worker(args, plan, state, persist):
    source = ROOT / "baselines/Play2Perfect"
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "rl_games"))
    # CUDA sees only physical GPU1, ordinal0. Vulkan still uses its physical
    # ordinal1, verified independently from Kit's active GPU/UUID table.
    import torch
    if torch.cuda.device_count() != 1: raise RuntimeError("Expected exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    from isaaclab.app import AppLauncher
    state["phase"] = "Kit startup"
    persist()
    portable = args.runtime / "kit-portable"
    portable.mkdir(exist_ok=True)
    kit_args = f"--portable-root {portable} --/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false --/app/settings/loadUserConfig=false"
    app = AppLauncher(device="cuda:0", multi_gpu=False, headless=True, enable_cameras=False, kit_args=kit_args).app
    try:
        _run_with_app(args, plan, state, persist, app, kit_args, portable)
    except BaseException as error:
        # Kit shutdown may terminate Python with exit0; retain failure BEFORE it.
        state.update(status="failed", exception={"type": type(error).__name__, "message": str(error)})
        persist()
        raise
    finally:
        state["phase"] = "Kit teardown"
        persist()
        app.close()


def _run_with_app(args, plan, state, persist, app, kit_args, portable):
    # Include renderer checks and imports in the App owner's cleanup scope.
    import carb.settings
    settings = carb.settings.get_settings()
    selection = {"renderer_active_gpu": settings.get("/renderer/activeGpu"),
                 "renderer_multi_gpu": settings.get("/renderer/multiGpu/enabled"),
                 "AppLauncher_device": "cuda:0", "selected_gpu_uuid": plan["selected_gpu_uuid"], "kit_args": kit_args,
                 "Kit_log_root": str(portable),
                 "verification_scope": "active GPU settings checked; review GPU Foundation table in retained Kit logs for physical Vulkan mapping"}
    atomic_json(args.output / "native_gpu_selection.json", selection)
    if selection["renderer_active_gpu"] != 1 or selection["renderer_multi_gpu"] is not False:
        raise RuntimeError("Kit did not retain explicitly selected renderer GPU1/multi-GPU-disabled settings")
    logs = list((portable / "logs").rglob("*.log"))
    if not logs: raise RuntimeError("Missing Kit log for physical Vulkan GPU verification")
    selection["physical_vulkan_verification"] = verify_vulkan_gpu_log(
        max(logs, key=lambda p: p.stat().st_mtime), plan["selected_gpu_uuid"], expected_pci_bus_id=plan["selected_gpu_pci_bus_id"])
    atomic_json(args.output / "native_gpu_selection.json", selection)
    state["phase"] = "native environment setup"
    persist()
    import math
    import numpy as np
    import torch
    import isaacsimenvs  # original task registration, retains native in-hand mechanism
    from isaaclab.sim import SimulationContext
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner
    from evaluation.eval_isaacsim import _apply_env_overrides, _configure_agent, _instantiate_env, _load_env_cfg

    counts = state["physics"]
    phase = "setup_physics_step_calls"
    original_step = SimulationContext.step
    def counted_step(self, *positional, **keywords):
        state["in_flight_physics_step"] = True
        persist()
        result = original_step(self, *positional, **keywords)
        counts[phase] += 1
        state["in_flight_physics_step"] = False
        persist()
        return result
    SimulationContext.step = counted_step
    env = None
    started = time.monotonic()
    try:
        cfg = _load_env_cfg("Isaacsimenvs-PreciseAssembly-Direct-v0")
        cfg.seed = args.seed
        _apply_env_overrides(cfg, problem="tight_insertion", goal_mode="preInsertAndFinal", random_goal_fraction=0.,
            insertion_success_tolerance=.01, retract_success_tolerance=.005, num_envs=args.num_envs,
            sim_device="cuda:0", sdf=False, keep_dr=False, extra_overrides={})
        env = _instantiate_env("Isaacsimenvs-PreciseAssembly-Direct-v0", cfg)
        if torch.device(env.device) != torch.device("cuda:0"):
            raise RuntimeError("Native environment did not establish the explicitly selected GPU")
        agent_cfg = _configure_agent("Isaacsimenvs-PreciseAssembly-Direct-v0", "rl_games_sapg_cfg_entry_point",
            rl_device="cuda:0", num_envs=args.num_envs, deterministic=True, games=10**9, extra_overrides={})
        agent_cfg["params"]["seed"] = args.seed
        wrapped = register_rlgames_env(env, rl_device="cuda:0",
            clip_obs=float(agent_cfg["params"]["env"].get("clip_observations", math.inf)),
            clip_actions=float(agent_cfg["params"]["env"].get("clip_actions", math.inf)))
        runner = Runner()
        runner.load(agent_cfg)
        runner.reset()
        player = runner.create_player()
        if next(player.model.parameters()).device != torch.device("cuda:0"):
            raise RuntimeError("Native actor did not establish the explicitly selected GPU")
        allowed = [np.core.multiarray.scalar, np.dtype, type(np.dtype("float32")), type(np.dtype("float64"))]
        with torch.serialization.safe_globals(allowed):
            weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        weights = weights[0] if 0 in weights else weights
        player.set_weights(weights)
        player.has_batch_dimension = True
        player.reset()
        if args.coefficient_mode == "exploitation": player.intr_reward_coef_embd.zero_()
        assignments = coefficient_assignment(player.intr_reward_coef_embd.flatten().cpu().tolist(),
                                             player.model.a2c_network.param_ids.flatten().cpu().tolist())
        state["coefficient_assignment"] = assignments
        persist()
        if not all(row["exact_match"] for row in assignments):
            raise RuntimeError("Actual native slot coefficient does not exactly match a trained parameter ID")
        if assignments != plan["coefficient_assignment"]:
            raise RuntimeError("Actual native coefficient assignment differs from reviewed plan")
        obs = player.env_reset(wrapped)
        rows = state["first_episode_rows"]
        for row in rows: row["status"] = "running"
        done_once = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        step_count = 0
        phase = "rollout_physics_step_calls"
        state["phase"] = "native rollout"
        persist()
        for step in range(args.steps):
            action = player.get_action(obs, is_deterministic=True)
            if not bool(torch.isfinite(action).all()): raise RuntimeError("Nonfinite published native actor action")
            state["in_flight_control_step"] = True
            persist()
            obs, _, dones, infos = player.env_step(wrapped, action)
            state["control_calls"] += 1
            state["in_flight_control_step"] = False
            # Every first episode accrues steps even when no slot has ended yet.
            for row in rows:
                if not row["ended"]: row["first_episode_steps"] = step + 1
            done = dones.reshape(-1).to(env.device).bool()
            newly = done & ~done_once
            retract_enabled = bool(cfg.precise_assembly.enable_retract)
            terminal = infos.get("episode_final", {})
            terminal_retract = terminal.get("retract_success")
            for index in torch.nonzero(newly, as_tuple=False).reshape(-1).tolist():
                goals = int(env.prev_episode_env_max_goals[index])
                achieved = int(env._prev_episode_successes[index])
                rows[index].update(ended=True, censored=False, status="ended", first_episode_steps=step + 1,
                    **native_fixed_goal_outcome(goals, achieved, retract_enabled=retract_enabled,
                        terminal_retract_success=(None if terminal_retract is None else bool(terminal_retract[index] > 0.5))),
                    source_would_filter_short_init=(step + 1 < 100))
            done_once |= newly
            state["completed_trials"] = int(done_once.sum())
            persist()
            # Match native player episode reset semantics for automatically reset slots.
            if player.states is not None:
                for rnn_state in player.states: rnn_state[:, done.to(rnn_state.device), :] = 0
            step_count = step + 1
            if bool(done_once.all()): break
        for row in rows:
            if not row["ended"]: row["first_episode_steps"] = step_count
        report = {**plan, "scope": "native published checkpoint first episodes; no training or original aggregate paper-score claim",
            **full_denominator_summary(rows, args.num_envs),
            "trials": rows, "source_filtered_rows_retained": True,
            "actual_control_calls": step_count, "actual_environment_control_transitions": step_count * args.num_envs,
            "coefficient_assignment_actual": assignments,
            "control_dt": float(env.step_dt), "physics": counts, "Kit_gpu_selection": selection,
            "physics_world_advances_from_instrumented_calls": sum(counts.values()) * args.num_envs,
            "accounting_scope": "Confirmed SimulationContext.step calls during setup/reset/rollout and all slots, including extra auto-reset episodes. Other internal Kit app.update startup ticks are outside this counter; wall/resource watchdog includes them.",
            "env_device_actual": str(env.device), "actor_device_actual": str(next(player.model.parameters()).device),
            "wall_seconds": time.monotonic() - started,
            "gpu_inventory": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], text=True, timeout=5)}
        (args.output / "native_closed_loop.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: report[key] for key in ("requested_trials", "completed_trials", "successes", "full_denominator_success_rate")}, indent=2), flush=True)
        state.update(status="rollout_completed", completed_trials=sum(row["ended"] for row in rows), phase="Kit teardown")
        persist()
    finally:
        SimulationContext.step = original_step
        if env is not None: env.close()


if __name__ == "__main__":
    main()
