"""Deterministic PHP BlinkMimic playback with the PHP reference ghost."""
import argparse
import time
import threading


SKILL_LABELS = {
    "random_locomotion": "随机行走与跑动",
    "low_speed_step": "低速上台阶",
    "high_speed_step": "高速上台阶",
    "low_speed_climb_58": "低速攀爬 58",
    "high_speed_climb_58": "高速攀爬 58",
    "low_speed_climb_76": "低速攀爬 76",
    "high_speed_climb_76": "高速攀爬 76",
    "low_speed_climb_94": "低速攀爬 94",
    "high_speed_climb_94": "高速攀爬 94",
    "low_speed_roll_78": "低速翻滚 78",
    "low_speed_roll_90": "低速翻滚 90",
    "high_speed_roll_134": "高速翻滚 134",
    "high_speed_hurdle": "高速跨栏",
    "high_speed_dash_vault": "高速冲刺翻越",
    "high_speed_speed_vault": "高速 Speed Vault",
    "high_speed_cat_vault": "高速猫扑翻越",
    "high_speed_jump_35": "高速跳跃 35",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-dir', default='data/php_motion')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--motion-id', type=int, default=0)
    parser.add_argument('--terrain-variant', type=int, default=0)
    parser.add_argument('--viewer', choices=('viser', 'none'), default='viser')
    parser.add_argument('--max-steps', type=int, default=0)
    args = parser.parse_args()
    if args.viewer == 'none' and args.max_steps <= 0:
        parser.error('--viewer none requires --max-steps > 0')

    import torch
    import blinkmimic.networks as networks
    from blinkmimic.robot import get_robot
    from blinkmimic.kinematics import Kinematics
    from blinkmimic.motion import MotionLib, CONTROL_DT
    from blinkmimic.dr import XiLayout
    from blinkmimic.mdp import make_mdp
    from blinkmimic.env import TrackingSim, make_sim
    from blinkmimic.networks import Policy, policy_kwargs
    from blinkmimic.php.checkpoint import validate

    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ck['args']
    from blinkmimic.php.config import configure
    robot = configure(get_robot('php'),args.dataset_dir,config=config)
    kin = Kinematics(robot, args.device)
    motion = MotionLib.from_dir(args.dataset_dir, robot, kin, args.device)
    mdp = make_mdp(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=args.device),
                   XiLayout(robot, args.device).dim, robot.sensing)
    K, H = config['goal_frames'], config['hist']
    validate(config, robot, motion, mdp, motion.goal_dim(K))
    networks.AMP_DTYPE = torch.bfloat16 if config.get('amp') == 'bf16' else None
    widths = lambda key: tuple(map(int, config[key].split(',')))
    pi = Policy(mdp.obs_dim, motion.goal_dim(K), robot.num_joints, hist=H,
                enc_hidden=widths('enc_hidden'), dec_hidden=widths('dec_hidden'),
                **policy_kwargs(config, robot)).to(args.device).eval()
    pi.load_state_dict(ck['pi'])
    if not 0 <= args.motion_id < len(motion.names):
        parser.error('motion-id is outside the loaded library')
    if not 0 <= args.terrain_variant < int(motion.variant_counts[args.motion_id]):
        parser.error('terrain-variant is unavailable for this motion')
    sim_cfg = make_sim(robot)
    if robot.terrain_format == "vertices":
        sim_cfg.nconmax = max(sim_cfg.nconmax, 512)
    env = TrackingSim(sim_cfg, 1, robot, motion, mdp, args.device, eval_mode=True)
    ids = torch.zeros(1, dtype=torch.long, device=args.device)
    mask = torch.ones(1, dtype=torch.bool, device=args.device)
    restart = threading.Event()
    server = None
    active, variant = args.motion_id, args.terrain_variant

    def reset():
        env.done_eval.zero_(); env.failed_eval.zero_()
        env.reset(ids, motion.clip_start[active:active+1])
        env.set_terrain(mask, env.f, torch.full_like(ids, variant))
        return env.observe().unsqueeze(1).repeat(1, H, 1)

    with torch.inference_mode():
        history = reset()
        if args.viewer == 'viser':
            import viser
            from supermimic_parkour.viewer import ComparisonScene
            server = viser.ViserServer(host=args.host, port=args.port, label='PHP BlinkMimic')
            comparison = ComparisonScene(server, env)
            server.gui.add_markdown(f'**PHP BlinkMimic** · iteration {ck["iteration"]}\n\n实体：策略 · 绿色：参考动作')
            skill_labels = [SKILL_LABELS.get(name, name) for name in motion.skill_names]
            clip_skills = [skill_labels[i] for i in motion.clip_skill.tolist()]
            clip_labels = [f"{i}: {name.rsplit('/', 1)[-1]}" for i, name in enumerate(motion.names)]
            clips_by_skill = {label: {} for label in skill_labels}
            for i, label in enumerate(clip_labels):
                clips_by_skill[clip_skills[i]][label] = i
            variant_counts = motion.variant_counts.tolist()
            with server.gui.add_folder("动作选择"):
                skill = server.gui.add_dropdown(
                    "技能种类",
                    options=tuple(label for label in skill_labels if clips_by_skill[label]),
                    initial_value=clip_skills[active],
                )
                selector = server.gui.add_dropdown(
                    "动作变体",
                    options=tuple(clips_by_skill[clip_skills[active]]),
                    initial_value=clip_labels[active],
                )
                terrain = server.gui.add_dropdown(
                    "地形变体",
                    options=tuple(str(i) for i in range(variant_counts[active])),
                    initial_value=str(variant),
                )
            playing = server.gui.add_checkbox('播放', initial_value=True)
            looping = server.gui.add_checkbox('结束后重播', initial_value=True)
            ghost = server.gui.add_checkbox('显示参考', initial_value=True)
            status = server.gui.add_markdown('')
            button = server.gui.add_button('重新开始')
            button.on_click(lambda _: restart.set())
            comparison.scene.create_scene_gui(camera_distance=4., camera_azimuth=135., camera_elevation=25., show_debug_viz_control=False)
            print(f'Open http://{args.host}:{server.get_port()}', flush=True)
        try:
            steps = 0
            while args.max_steps == 0 or steps < args.max_steps:
                t0 = time.monotonic()
                if server:
                    selected_skill = skill.value
                    choices = clips_by_skill[selected_skill]
                    with server.atomic():
                        if selected_skill != clip_skills[active]:
                            selector.options = tuple(choices)
                        # A client can still submit a choice from the previous skill.
                        chosen_label = selector.value
                        if chosen_label not in choices:
                            chosen_label = next(iter(choices))
                            selector.value = chosen_label
                        selected = choices[chosen_label]
                        if selected != active:
                            active = selected
                            terrain.options = tuple(str(i) for i in range(variant_counts[active]))
                            terrain.value = "0"
                            restart.set()
                        chosen_terrain = terrain.value
                        if chosen_terrain not in terrain.options:
                            chosen_terrain = "0"
                            terrain.value = chosen_terrain
                        if int(chosen_terrain) != variant:
                            variant = int(chosen_terrain)
                            restart.set()
                if restart.is_set():
                    history = reset(); restart.clear()
                done = bool(env.done_eval[0])
                if done and (not server or looping.value):
                    history = reset(); done = False
                if not done and (not server or playing.value):
                    o = env.observe()
                    history = torch.cat((history[:, 1:], o.unsqueeze(1)), 1)
                    env.step(pi(history, motion.goal(env.f, K)))
                    steps += 1
                if server:
                    comparison.sync(show_reference=ghost.value)
                    status.content = (
                        f'**{clip_skills[active]}** · 地形 {variant}\n\n'
                        f'{clip_labels[active]} · frame {int(env.f[0]-motion.clip_start[active])}'
                        f' · failed={bool(env.failed_eval[0])}'
                    )
                    time.sleep(max(0, CONTROL_DT-(time.monotonic()-t0)))
            print(f'Playback verified: {steps} simulation steps; finite={bool(torch.isfinite(env.state()[0]).all())}', flush=True)
        finally:
            if server:
                server.stop()


if __name__ == '__main__':
    main()
