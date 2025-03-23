"""Utility functions to save rendering videos."""

#from __future__ import annotations

import os
import re
import subprocess
import pickle

import gymnasium as gym

import cv2
import torch

def record(
    seed: int,
    environment: gym.Env,
    replay_actions_dir: str,
    videos_dir: str,
    name_prefix: str,
    #episode_index: int,
    actions: torch.Tensor,
    **kwargs,
):
    """Save videos from rendering frames.

    This function extract video from a list of render frame episodes.

    Args:
        frames (List[RenderFrame]): A list of frames to compose the video.
        video_folder (str): The folder where the recordings will be stored
        episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
        step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
        video_length (int): The length of recorded episodes. If it isn't specified, the entire episode is recorded.
            Otherwise, snippets of the specified length are captured.
        name_prefix (str): Will be prepended to the filename of the recordings.
        episode_index (int): The index of the current episode.
        step_starting_index (int): The step index of the first frame.
        save_logger: If to log the video saving progress, helpful for long videos that take a while, use "bar" to enable.
        **kwargs: The kwargs that will be passed to moviepy's ImageSequenceClip.
            You need to specify either fps or duration.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.utils.save_video import save_video
        >>> env = gym.make("FrozenLake-v1", render_mode="rgb_array_list")
        >>> _ = env.reset()
        >>> step_starting_index = 0
        >>> episode_index = 0
        >>> for step_index in range(199): # doctest: +SKIP
        ...    action = env.action_space.sample()
        ...    _, _, terminated, truncated, _ = env.step(action)
        ...
        ...    if terminated or truncated:
        ...       save_video(
        ...          frames=env.render(),
        ...          video_folder="videos",
        ...          fps=env.metadata["render_fps"],
        ...          step_starting_index=step_starting_index,
        ...          episode_index=episode_index
        ...       )
        ...       step_starting_index = step_index + 1
        ...       episode_index += 1
        ...       env.reset()
        >>> env.close()
    """
    videos_dir = os.path.abspath(videos_dir)
    os.makedirs(videos_dir, exist_ok=True)
    replay_actions_dir = os.path.abspath(replay_actions_dir)
    os.makedirs(replay_actions_dir, exist_ok=True)

    name_prefix = name_prefix.replace("/", "-")
    example_name = make_example_name(videos_dir, replay_actions_dir, name_prefix, seed)
    replay_action_path = f"{replay_actions_dir}/{example_name}.pkl"
    video_path = f"{videos_dir}/{example_name}.mp4"
    fps = environment.metadata["render_fps"]
    scale_factor = kwargs["scale_factor"]
    speed = kwargs["speed"]

    observation, _ = environment.reset(seed=seed)
    frame_height, frame_width, _ = observation.shape
    new_frame_height = int(scale_factor * frame_height)
    new_frame_width = int(scale_factor * frame_width)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # H.264: x264, AV1: av01
    video = cv2.VideoWriter(video_path, fourcc, fps*speed, (new_frame_width, new_frame_height))

    # Save the action sequence
    with open(f"{replay_action_path}.pkl", 'wb') as f:
        pickle.dump(actions, f)

    actions = actions[0, :, 0, :]

    for action in actions:
        observation, _, _, _, _ = environment.step(action)
        print(observation.shape)
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (new_frame_width, new_frame_height))
        #frame = frame.astype(np.uint8)
        video.write(frame)

    video.release()

    """ffmpeg_command = [
        'ffmpeg',  # FFmpeg 실행 파일을 호출
        '-y',  # 기존 파일이 있으면 덮어쓰기
        '-f', 'rawvideo',  # 입력 데이터의 포맷을 'rawvideo'로 지정 (압축되지 않은 비디오 데이터)
        '-vcodec', 'libx264',  # 비디오 코덱을 'rawvideo'로 지정 (압축하지 않음) # libx264: H.264, libaom-av1: AV1
        '-pix_fmt', 'bgr24',  # 색상 포맷을 'bgr24'로 지정 (OpenCV에서 사용하는 기본 색상 포맷)
        '-s', f'{frame_width}x{frame_height}',  # 비디오의 해상도를 지정 (예: 1920x1080)
        '-r', str(fps),  # 초당 프레임 수 (fps) 설정
        '-i', '-',  # 입력 파일로 '표준 입력'을 사용 ('-'는 표준 입력을 의미)
        video_path  # 출력 파일 경로 (영상이 저장될 파일 경로)
        #'-b:v', '1M'  # 비트레이트 1Mbps 설정, 높은 비트레이트일수록 더 좋은 품질을 제공하지만 파일 크기가 커짐
        #'-crf', '23'  # 품질을 조정하는 CRF 값, 0은 최고 품질, 51은 최저 품질, 일반적으로 18~28에서 사용
        #'-cpu-used', '4'  # 중간 정도의 속도와 품질 0~8
    ]
    # FFmpeg 프로세스 시작
    process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
    
    # 예시 NumPy 배열로 비디오 프레임 만들기
    for frame in observations:
        frame = frame.astype(np.uint8)  # 프레임을 uint8 형식으로 변환 (0-255 범위)
        process.stdin.write(frame.tobytes())  # 배열을 바이트로 변환하여 FFmpeg에 전달

    process.stdin.close()
    process.wait()""" #공부필요

    print(f"Replay Action saved as {replay_action_path}")
    print(f"Video saved as {video_path}")

def make_example_name(replay_actions_dir, videos_dir, model_name, seed):
    replay_actions_examples_numbers = [
        int(match.group(1)) for f in os.listdir(replay_actions_dir)
        if (match := re.search(rf"{model_name}.*?_episode_(\d+)", f))
    ] or [0]
    video_examples_numbers = [
        int(match.group(1)) for f in os.listdir(videos_dir)
        if (match := re.search(rf"{model_name}.*?_episode_(\d+)", f))
    ] or [0]

    example_number = max(max(replay_actions_examples_numbers), max(video_examples_numbers)) + 1
    return f"{model_name}_{seed}_episode_{example_number}"