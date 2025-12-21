import os 
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 작성해주신 utils.py와 model.py에서 클래스들을 불러옵니다.
from utils import * 
from model import * 
# ==========================================
# 1. 설정 (Configuration)
# ==========================================
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 256        # 메모리 부족 시 64로 조절
    LR = 0.001
    EPOCHS = 50
    NUM_WORKERS = 4
    
    # 데이터 정규화 상수
    MAX_X = 105.0
    MAX_Y = 68.0
    MAX_TIME = 5700.0       # 95분
    EOS_VALUE = -1.0
    
    # 임베딩 관련 설정
    NUM_ACTIONS = 33        # Action 종류 개수
    MAX_PHASE_LEN_EMBED = 30 # Phase 길이 임베딩 최대값
    ACTION_EMB_DIM = 4      # 매 스텝 Action 임베딩 차원
    LEN_EMB_DIM = 4         # Length 임베딩 차원
    
    DOMAIN_INPUT_DIM = 2    # [NEW] 도메인 피처 차원 (dx, dy, ddist, speed, norm_len)
    
    # 모델 하이퍼파라미터
    INPUT_SIZE = 5          # [sx, sy, ex, ey, t]
    PHASE_HIDDEN = 32
    EPISODE_HIDDEN = 128
    DROPOUT = 0.4
    
    # 경로
    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    WEIGHT_DIR = './weight'

print(f"✅ 사용 장치: {Config.DEVICE}")
if torch.cuda.is_available():
    print(f"   - GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. Training Engine
# ==========================================
def run_training():
    os.makedirs(Config.WEIGHT_DIR, exist_ok=True)
    
    print("📂 데이터 로드 중...")
    # utils.py에 정의된 Complete 데이터셋 사용
    train_dataset = SoccerCompleteDataset(Config.TRAIN_DIR)
    val_dataset = SoccerCompleteDataset(Config.VAL_DIR)
    
    # utils.py에 정의된 complete_collate_fn 사용
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, collate_fn=complete_collate_fn, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, collate_fn=complete_collate_fn, 
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    print(f"   - Train Files: {len(train_dataset)}")
    print(f"   - Val Files: {len(val_dataset)}")

    # 모델 초기화 (CompleteHierarchicalLSTM)
    model = CompleteHierarchicalLSTM(
        input_size=Config.INPUT_SIZE,
        phase_hidden=Config.PHASE_HIDDEN,
        episode_hidden=Config.EPISODE_HIDDEN,
        output_size=2,
        dropout=Config.DROPOUT,
        num_actions=Config.NUM_ACTIONS,
        action_emb_dim=Config.ACTION_EMB_DIM,
        max_phase_len=Config.MAX_PHASE_LEN_EMBED,
        len_emb_dim=Config.LEN_EMB_DIM,
        domain_input_dim=Config.DOMAIN_INPUT_DIM  # [NEW]
    ).to(Config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.MSELoss()
    
    best_dist_error = float('inf')
    
    # --- Epoch Loop ---
    for epoch in range(Config.EPOCHS):
        print(f"\nTraining Epoch {epoch+1}/{Config.EPOCHS}")
        
        # 1. Train
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="[Train]"):
            # Unpack 7 Items (순서 중요: collate_fn 반환 순서와 일치)
            padded_phases, padded_actions, phase_lengths, episode_lengths, targets, phase_len_ids, domain_features = batch
            
            if padded_phases is None: continue
            
            # Move to Device (모든 텐서를 GPU로 이동)
            padded_phases = padded_phases.to(Config.DEVICE)
            padded_actions = padded_actions.to(Config.DEVICE) # [NEW]
            phase_lengths = phase_lengths.to(Config.DEVICE)
            episode_lengths = episode_lengths.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            phase_len_ids = phase_len_ids.to(Config.DEVICE)
            domain_features = domain_features.to(Config.DEVICE) # [NEW]
            
            optimizer.zero_grad()
            
            # Forward (인자 순서 확인: model.forward 정의와 일치)
            preds = model(padded_phases, padded_actions, phase_lengths, episode_lengths, phase_len_ids, domain_features)
            
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 2. Validation
        model.eval()
        val_loss = 0.0
        total_dist_error = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Valid]"):
                padded_phases, padded_actions, phase_lengths, episode_lengths, targets, phase_len_ids, domain_features = batch
                
                if padded_phases is None: continue
                
                # Move to Device
                padded_phases = padded_phases.to(Config.DEVICE)
                padded_actions = padded_actions.to(Config.DEVICE)
                phase_lengths = phase_lengths.to(Config.DEVICE)
                episode_lengths = episode_lengths.to(Config.DEVICE)
                targets = targets.to(Config.DEVICE)
                phase_len_ids = phase_len_ids.to(Config.DEVICE)
                domain_features = domain_features.to(Config.DEVICE)
                
                # Forward
                preds = model(padded_phases, padded_actions, phase_lengths, episode_lengths, phase_len_ids, domain_features)
                
                loss = criterion(preds, targets)
                val_loss += loss.item()
                
                # 거리 오차 계산 (Meter)
                pred_real_x = preds[:, 0] * Config.MAX_X
                pred_real_y = preds[:, 1] * Config.MAX_Y
                true_real_x = targets[:, 0] * Config.MAX_X
                true_real_y = targets[:, 1] * Config.MAX_Y
                
                dist = torch.sqrt((pred_real_x - true_real_x)**2 + (pred_real_y - true_real_y)**2)
                total_dist_error += dist.sum().item()
                count += targets.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_dist_error = total_dist_error / count if count > 0 else 0.0
        
        print(f"   Results: Train Loss {avg_train_loss:.5f} | Val Loss {avg_val_loss:.5f}")
        print(f"   📏 Avg Distance Error: {avg_dist_error:.4f} meters")
        
        # Best Model 저장
        # if avg_dist_error < best_dist_error:
        #     best_dist_error = avg_dist_error
        #     # 파일명에 complete 명시
        #     save_name = f"complete_hierarchical_lr{Config.LR}_dist{best_dist_error:.4f}m.pth"
        #     save_path = os.path.join(Config.WEIGHT_DIR, save_name)
        #     torch.save(model.state_dict(), save_path)
        #     print(f"   💾 Best Model Saved: {save_name}")

if __name__ == "__main__":
    run_training()