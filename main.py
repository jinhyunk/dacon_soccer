import os 
import tqdm 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import * 
from model import * 

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 128        # 3060 메모리 활용 (메모리 부족 시 64로 조절)
    LR = 0.001
    EPOCHS = 50
    NUM_WORKERS = 4         # 데이터 로딩 병렬 처리 개수
    
    # 데이터 정규화 상수
    MAX_X = 105.0
    MAX_Y = 68.0
    MAX_TIME = 5700.0       # 95분
    EOS_VALUE = -1.0
    
    # 모델 하이퍼파라미터
    INPUT_SIZE = 5          # [sx, sy, ex, ey, t]
    PHASE_HIDDEN = 64       # Phase LSTM의 Hidden Size
    EPISODE_HIDDEN = 256    # Episode LSTM의 Hidden Size
    DROPOUT = 0.3
    
    # 경로
    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    WEIGHT_DIR = './weight'

print(f"✅ 사용 장치: {Config.DEVICE}")
if torch.cuda.is_available():
    print(f"   - GPU: {torch.cuda.get_device_name(0)}")


def run_training():
    # 저장 폴더 생성
    os.makedirs(Config.WEIGHT_DIR, exist_ok=True)
    
    # 데이터셋 로드
    print("📂 데이터 로드 중...")
    train_dataset = SoccerDataset(Config.TRAIN_DIR)
    val_dataset = SoccerDataset(Config.VAL_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, collate_fn=hierarchical_collate_fn, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, collate_fn=hierarchical_collate_fn, 
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    print(f"   - Train Files: {len(train_dataset)}")
    print(f"   - Val Files: {len(val_dataset)}")

    # 모델 초기화
    model = HierarchicalLSTM(
        input_size=Config.INPUT_SIZE,
        phase_hidden_size=Config.PHASE_HIDDEN,
        episode_hidden=Config.EPISODE_HIDDEN,
        dropout=Config.DROPOUT
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
            padded_phases, phase_lengths, episode_lengths, targets = batch
            
            if padded_phases is None: continue
            
            # GPU 이동
            padded_phases = padded_phases.to(Config.DEVICE)
            # lengths는 CPU에 둬도 pack_padded_sequence에서 처리가능하지만, 
            # 모델 내부에서 .cpu() 호출이 있으므로 여기선 그대로 둠.
            # targets는 GPU로
            targets = targets.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(padded_phases, phase_lengths, episode_lengths)
            
            # Loss
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
                padded_phases, phase_lengths, episode_lengths, targets = batch
                if padded_phases is None: continue
                
                padded_phases = padded_phases.to(Config.DEVICE)
                targets = targets.to(Config.DEVICE)
                
                preds = model(padded_phases, phase_lengths, episode_lengths)
                
                loss = criterion(preds, targets)
                val_loss += loss.item()
                
                # 거리 오차 계산 (Meter 단위)
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
        if avg_dist_error < best_dist_error:
            best_dist_error = avg_dist_error
            save_name = f"hierarchical_lr{Config.LR}_dist{best_dist_error:.4f}m.pth"
            save_path = os.path.join(Config.WEIGHT_DIR, save_name)
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Best Model Saved: {save_name}")

if __name__ == "__main__":
    run_training()