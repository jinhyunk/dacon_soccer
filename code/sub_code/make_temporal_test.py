import os

import numpy as np
import pandas as pd
from tqdm import tqdm


PHASE_TRAIN_PATH = "../../data/phase_train.csv"
TEMPORAL_TEST_ROOT = "../../data/temporal_test"
TRAIN_OUTPUT_PATH = "../../data/train.csv"
TEMPORAL_TEST_CSV_PATH = "../../data/temporal_test.csv"


def main(sample_frac: float = 0.1, random_state: int = 42) -> None:
    """
    phase_train.csv 에서 game_id 단위로 약 10%를 샘플링하여
    ../data/test 와 동일한 구조의 ../data/temporal_test 폴더를 생성한다.

    - game_id 별 디렉토리 생성
    - 각 디렉토리 안에 game_episode (예: 126283_1) 단위로 csv 파일 생성
      파일명: {game_episode}.csv

    또한, 10%를 제외한 나머지 90% 데이터를 ../data/train.csv 로 저장한다.
    train.csv 와 temporal_test 는 game_id 가 겹치지 않음.
    """
    os.makedirs(TEMPORAL_TEST_ROOT, exist_ok=True)

    # 원본 학습 데이터 로드 (수정 X)
    df = pd.read_csv(PHASE_TRAIN_PATH)

    # game_id 단위로 10% 샘플링 (겹치지 않도록)
    unique_game_ids = df["game_id"].unique()
    np.random.seed(random_state)
    sampled_game_ids = np.random.choice(
        unique_game_ids,
        size=int(len(unique_game_ids) * sample_frac),
        replace=False,
    )
    sampled_game_ids_set = set(sampled_game_ids)

    # temporal_test용 데이터 (샘플링된 game_id)
    df_sample = df[df["game_id"].isin(sampled_game_ids_set)].copy()
    df_sample = df_sample.sort_values(["game_id", "game_episode", "time_seconds"]).reset_index(drop=True)

    # train용 데이터 (나머지 game_id)
    df_train = df[~df["game_id"].isin(sampled_game_ids_set)].reset_index(drop=True)
    df_train.to_csv(TRAIN_OUTPUT_PATH, index=False)

    print(f"phase_train total rows: {len(df)}")
    print(f"  - unique game_ids: {len(unique_game_ids)}")
    print(f"  - sampled game_ids (temporal_test): {len(sampled_game_ids)} ({len(df_sample)} rows)")
    print(f"  - remaining game_ids (train): {len(unique_game_ids) - len(sampled_game_ids)} ({len(df_train)} rows)")
    print(f"Train data saved to: {TRAIN_OUTPUT_PATH}")

    # game_id 별 디렉토리 + game_episode 단위 csv 생성
    # temporal_test.csv 용 메타 정보 수집
    meta_rows = []

    for (game_id, game_episode), g in tqdm(
        df_sample.groupby(["game_id", "game_episode"]),
        desc="Writing temporal_test files",
    ):
        game_id = int(game_id)
        game_dir = os.path.join(TEMPORAL_TEST_ROOT, str(game_id))
        os.makedirs(game_dir, exist_ok=True)

        # 파일명은 test 구조와 최대한 유사하게: {game_episode}.csv
        # game_episode 는 이미 "126283_1" 같은 형태이므로 그대로 사용
        csv_name = f"{game_episode}.csv"
        csv_path = os.path.join(game_dir, csv_name)

        # 인덱스 컬럼은 굳이 넣지 않고, test 예시와 비슷하게 index=False 로 저장
        g.to_csv(csv_path, index=False)

        # 메타 정보 추가 (base_test.csv와 동일한 형식)
        relative_path = f"./temporal_test/{game_id}/{game_episode}.csv"
        meta_rows.append({
            "game_id": game_id,
            "game_episode": game_episode,
            "path": relative_path,
        })

    print(f"Temporal test data written under: {TEMPORAL_TEST_ROOT}")

    # temporal_test.csv 생성 (base_test.csv와 동일한 형식)
    meta_df = pd.DataFrame(meta_rows)
    meta_df = meta_df.sort_values(["game_id", "game_episode"]).reset_index(drop=True)
    meta_df.to_csv(TEMPORAL_TEST_CSV_PATH, index=False)
    print(f"Temporal test meta saved to: {TEMPORAL_TEST_CSV_PATH} ({len(meta_df)} episodes)")


if __name__ == "__main__":
    main()


