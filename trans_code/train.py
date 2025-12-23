"""
Transformer 좌표 예측 모델 학습 스크립트

Usage:
    # 기본 실행 (episode 모드, scratch 학습)
    python train.py
    
    # episode + phase 모드
    python train.py --data_mode=episode_phase
    
    # phase 모드 (phase별로 시퀀스 생성)
    python train.py --data_mode=phase
    
    # team_id 모드 ((game_episode, team_id) 별로 시퀀스 생성)
    python train.py --data_mode=team_id
    
    # phase pretrain + episode finetune 모드
    python train.py --data_mode=pretrain_finetune --pretrain_epochs=10 --finetune_epochs=20
    
    # team_id pretrain + episode finetune 모드
    python train.py --data_mode=pretrain_finetune --pretrain_mode=team_id
    
    # Team-wise 학습 (main_code 방식: pretrain -> 각 team별 finetune)
    python train.py --data_mode=team_wise --pretrain_mode=phase --pretrain_epochs=10 --finetune_epochs=20
    
    # 에폭 수 변경
    python train.py --epochs=50
    
    # 모델 타입 변경
    python train.py --model_type=pooling
    
    # Pretrained 모델 사용 (GPT-2)
    python train.py --model_type=pretrained_gpt2
    
    # Pretrained 모델 사용 (DistilBERT - 경량)
    python train.py --model_type=pretrained_distilbert
    
    # Pretrained 레이어 freeze (전이학습)
    python train.py --model_type=pretrained_gpt2 --freeze_pretrained
    
    # wandb 로깅 활성화
    python train.py --wandb --wandb_project=soccer_prediction
    
    # wandb run 이름 지정
    python train.py --wandb --wandb_project=soccer --wandb_run_name=exp_001
"""
import os
import argparse

import pandas as pd
import torch

from config import (
    TRAIN_PATH,
    MODEL_DIR,
    MODEL_PATH,
    EPOCHS,
    BATCH_SIZE,
    D_MODEL,
    NHEAD,
    NUM_LAYERS,
    DIM_FEEDFORWARD,
    DROPOUT,
    LR,
    log_device,
)
from dataset import build_train_sequences, build_dataloaders
from models import create_model
from trainer import train_loop, save_model


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description="Transformer 좌표 예측 모델 학습")
    
    # 데이터 모드
    parser.add_argument(
        "--data_mode",
        type=str,
        default="episode",
        choices=["episode", "episode_phase", "phase", "team_id", "pretrain_finetune", "team_wise"],
        help="데이터 모드: 'episode', 'episode_phase', 'phase', 'team_id', 'pretrain_finetune', 'team_wise'"
    )
    parser.add_argument(
        "--pretrain_mode",
        type=str,
        default="phase",
        choices=["phase", "team_id"],
        help="Pretrain 기준 (pretrain_finetune, team_wise 모드에서 사용): 'phase' 또는 'team_id'"
    )
    
    # 에폭 설정
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"학습 에폭 수 (pretrain_finetune 모드가 아닐 때 사용) (default: {EPOCHS})"
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=10,
        help="Pretrain 에폭 수 (pretrain_finetune 모드일 때만 사용) (default: 10)"
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=20,
        help="Finetune 에폭 수 (pretrain_finetune 모드일 때만 사용) (default: 20)"
    )
    
    # 기타 하이퍼파라미터
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"배치 크기 (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="encoder",
        choices=["encoder", "pooling", "pretrained_gpt2", "pretrained_bert", "pretrained_distilbert"],
        help="모델 타입: 'encoder', 'pooling', 'pretrained_gpt2', 'pretrained_bert', 'pretrained_distilbert'"
    )
    parser.add_argument(
        "--freeze_pretrained",
        action="store_true",
        help="Pretrained 모델 레이어 freeze (입출력 레이어만 학습)"
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default=None,
        help="Pretrained 모델 이름 (예: 'gpt2-medium', 'bert-large-uncased')"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=D_MODEL,
        help=f"모델 차원 (default: {D_MODEL})"
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=NHEAD,
        help=f"어텐션 헤드 수 (default: {NHEAD})"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=NUM_LAYERS,
        help=f"Transformer 레이어 수 (default: {NUM_LAYERS})"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"학습률 (default: {LR})"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="조기 종료 patience (default: 10)"
    )
    parser.add_argument(
        "--no_scheduler",
        action="store_true",
        help="학습률 스케줄러 비활성화"
    )
    
    # wandb 관련 인자
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="wandb 로깅 활성화"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="soccer_coord_prediction",
        help="wandb 프로젝트 이름 (default: soccer_coord_prediction)"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="wandb run 이름 (미지정 시 자동 생성)"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="wandb entity (팀/사용자 이름)"
    )
    
    return parser.parse_args()


def train_single_mode(args, df: pd.DataFrame, data_mode: str, epochs: int, desc: str, pretrain_state=None, use_wandb: bool = False):
    """
    단일 모드로 학습 수행
    
    Args:
        args: 커맨드라인 인자
        df: 학습 데이터
        data_mode: 'episode', 'episode_phase', 'phase' 중 하나
        epochs: 학습 에폭 수
        desc: 설명 prefix
        pretrain_state: pretrain된 가중치 (finetune용)
        use_wandb: wandb 로깅 사용 여부
    
    Returns:
        model, best_state, best_dist
    """
    print(f"\n========== Building Sequences ({data_mode}) ==========")
    episodes, targets = build_train_sequences(df, data_mode=data_mode)
    
    print(f"\n========== Creating DataLoaders ==========")
    train_loader, valid_loader = build_dataloaders(
        episodes, targets, batch_size=args.batch_size
    )
    
    print(f"\n========== Creating Model ==========")
    model, criterion, optimizer = create_model(
        model_type=args.model_type,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        lr=args.lr,
        pretrain_state=pretrain_state,
        freeze_pretrained=args.freeze_pretrained,
        pretrained_model_name=args.pretrained_model_name,
    )
    
    print(f"\n========== {desc} ==========")
    print(f"Data mode: {data_mode}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    model, best_state, best_dist = train_loop(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=epochs,
        use_scheduler=not args.no_scheduler,
        early_stop_patience=args.patience,
        desc_prefix=f"[{desc}]",
        use_wandb=use_wandb,
    )
    
    return model, best_state, best_dist


def init_wandb(args) -> bool:
    """wandb 초기화"""
    if not args.wandb:
        return False
    
    try:
        import wandb
    except ImportError:
        print("[WARN] wandb not installed. Run: pip install wandb")
        return False
    
    # wandb config 생성
    config = {
        "data_mode": args.data_mode,
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "patience": args.patience,
        "freeze_pretrained": args.freeze_pretrained,
        "pretrained_model_name": args.pretrained_model_name,
        "use_scheduler": not args.no_scheduler,
    }
    
    if args.data_mode == "pretrain_finetune":
        config["pretrain_epochs"] = args.pretrain_epochs
        config["finetune_epochs"] = args.finetune_epochs
    
    # run 이름 자동 생성 (미지정 시)
    run_name = args.wandb_run_name
    if run_name is None:
        run_name = f"{args.model_type}_{args.data_mode}_ep{args.epochs}"
    
    # wandb 초기화
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=config,
    )
    
    print(f"\n========== wandb Initialized ==========")
    print(f"Project: {args.wandb_project}")
    print(f"Run name: {run_name}")
    print(f"Run URL: {wandb.run.url}")
    
    return True


def main():
    args = parse_args()
    
    # Device 확인
    log_device()
    
    # 모델 저장 디렉토리 생성
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # wandb 초기화
    use_wandb = init_wandb(args)
    
    # =========================
    # 데이터 로드
    # =========================
    print(f"\n========== Loading Data ==========")
    print(f"Train path: {TRAIN_PATH}")
    
    df = pd.read_csv(TRAIN_PATH)
    print(f"Total rows: {len(df):,}")
    print(f"Unique game_episodes: {df['game_episode'].nunique():,}")
    print(f"Unique phases: {df['phase'].nunique():,}")
    
    # =========================
    # 학습 모드에 따른 분기
    # =========================
    
    if args.data_mode == "pretrain_finetune":
        # ========== Pretrain + Episode Finetune ==========
        print(f"\n{'='*50}")
        print(f"Mode: Pretrain ({args.pretrain_mode}) -> Finetune (episode)")
        print(f"{'='*50}")
        
        # Step 1: Pretrain (phase 또는 team_id)
        _, pretrain_state, pretrain_dist = train_single_mode(
            args=args,
            df=df,
            data_mode=args.pretrain_mode,
            epochs=args.pretrain_epochs,
            desc=f"Pretrain ({args.pretrain_mode})",
            pretrain_state=None,
            use_wandb=use_wandb,
        )
        print(f"\nPretrain finished. Best dist: {pretrain_dist:.4f}")
        
        # Pretrain 모델 저장
        pretrain_path = MODEL_PATH.replace(".pth", "_pretrain.pth")
        save_model(pretrain_state, pretrain_path)
        
        # Step 2: Episode로 Finetune (pretrain 가중치 사용)
        _, best_state, best_dist = train_single_mode(
            args=args,
            df=df,
            data_mode="episode",
            epochs=args.finetune_epochs,
            desc="Finetune (episode)",
            pretrain_state=pretrain_state,
            use_wandb=use_wandb,
        )
        final_model_path = MODEL_PATH.replace(".pth", "_pretrain_finetune.pth")
        
    elif args.data_mode == "team_wise":
        # ========== Team-wise Training (Pretrain + per-team Finetune) ==========
        print(f"\n{'='*50}")
        print(f"Mode: Pretrain ({args.pretrain_mode}) -> Team-wise Finetune")
        print(f"{'='*50}")
        
        # Step 1: Pretrain (phase 또는 team_id)
        _, pretrain_state, pretrain_dist = train_single_mode(
            args=args,
            df=df,
            data_mode=args.pretrain_mode,
            epochs=args.pretrain_epochs,
            desc=f"Pretrain ({args.pretrain_mode})",
            pretrain_state=None,
            use_wandb=use_wandb,
        )
        print(f"\nPretrain finished. Best dist: {pretrain_dist:.4f}")
        
        # Pretrain 모델 저장
        pretrain_path = MODEL_PATH.replace(".pth", "_pretrain.pth")
        save_model(pretrain_state, pretrain_path)
        
        # Step 2: Team별 Finetune
        team_model_dir = MODEL_DIR.replace("trans_models", "trans_team_models")
        os.makedirs(team_model_dir, exist_ok=True)
        
        team_ids = sorted(df["team_id"].unique())
        print(f"\nFound {len(team_ids)} teams: {team_ids}")
        
        for team_id in team_ids:
            try:
                from dataset import build_episode_team_sequences
                episodes, targets = build_episode_team_sequences(df, int(team_id))
                
                if len(episodes) < 2:
                    print(f"[team {team_id}] Skip: not enough data")
                    continue
                
                train_loader, valid_loader = build_dataloaders(
                    episodes, targets, batch_size=args.batch_size
                )
                
                model, criterion, optimizer = create_model(
                    model_type=args.model_type,
                    d_model=args.d_model,
                    nhead=args.nhead,
                    num_layers=args.num_layers,
                    dim_feedforward=DIM_FEEDFORWARD,
                    dropout=DROPOUT,
                    lr=args.lr,
                    pretrain_state=pretrain_state,
                    freeze_pretrained=args.freeze_pretrained,
                    pretrained_model_name=args.pretrained_model_name,
                )
                
                _, team_best_state, team_best_dist = train_loop(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    epochs=args.finetune_epochs,
                    use_scheduler=not args.no_scheduler,
                    early_stop_patience=args.patience,
                    desc_prefix=f"[team {team_id}]",
                    use_wandb=use_wandb,
                )
                
                # 팀별 모델 저장
                team_model_path = os.path.join(team_model_dir, f"team_{int(team_id)}.pth")
                save_model(team_best_state, team_model_path)
                print(f"[team {team_id}] Best dist: {team_best_dist:.4f}")
                
            except Exception as e:
                print(f"[team {team_id}] Error: {e}")
                continue
        
        # team_wise 모드의 경우 최종 모델 경로는 team_model_dir
        final_model_path = team_model_dir
        best_state = pretrain_state  # pretrain 모델을 대표로
        best_dist = pretrain_dist
        
    else:
        # ========== Single Mode Training ==========
        _, best_state, best_dist = train_single_mode(
            args=args,
            df=df,
            data_mode=args.data_mode,
            epochs=args.epochs,
            desc=f"Training ({args.data_mode})",
            pretrain_state=None,
            use_wandb=use_wandb,
        )
        
        # 모델 경로 결정
        if args.data_mode == "episode":
            final_model_path = MODEL_PATH
        elif args.data_mode == "episode_phase":
            final_model_path = MODEL_PATH.replace(".pth", "_epi_phase.pth")
        elif args.data_mode == "phase":
            final_model_path = MODEL_PATH.replace(".pth", "_phase.pth")
        elif args.data_mode == "team_id":
            final_model_path = MODEL_PATH.replace(".pth", "_team_id.pth")
        else:
            final_model_path = MODEL_PATH.replace(".pth", f"_{args.data_mode}.pth")
    
    # =========================
    # 최종 모델 저장
    # =========================
    print(f"\n========== Saving Model ==========")
    
    if args.data_mode != "team_wise":
        save_model(best_state, final_model_path)
        
        # data_mode 정보를 별도 파일로 저장 (inference에서 사용)
        mode_info_path = final_model_path.replace(".pth", "_mode.txt")
        with open(mode_info_path, "w") as f:
            f.write(args.data_mode)
        print(f"Data mode info saved to: {mode_info_path}")
    else:
        # team_wise 모드는 모드 정보를 디렉토리에 저장
        mode_info_path = os.path.join(final_model_path, "mode.txt")
        with open(mode_info_path, "w") as f:
            f.write("team_wise")
        print(f"Team-wise models saved to: {final_model_path}")
    
    print(f"\n========== Training Complete ==========")
    print(f"Data mode: {args.data_mode}")
    print(f"Best validation distance: {best_dist:.4f}")
    print(f"Model saved to: {final_model_path}")
    
    # wandb 종료 및 모델 artifact 저장
    if use_wandb:
        try:
            import wandb
            
            # 최종 결과 기록
            wandb.run.summary["final_best_dist"] = best_dist
            wandb.run.summary["model_path"] = final_model_path
            
            # 모델 artifact 저장 (선택적)
            artifact = wandb.Artifact(
                name=f"model_{args.model_type}_{args.data_mode}",
                type="model",
                description=f"Best model with dist={best_dist:.4f}"
            )
            artifact.add_file(final_model_path)
            wandb.log_artifact(artifact)
            
            wandb.finish()
            print("wandb run finished and model artifact saved.")
        except Exception as e:
            print(f"[WARN] wandb finish error: {e}")


if __name__ == "__main__":
    main()
