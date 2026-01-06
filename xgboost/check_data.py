"""데이터 구조 확인 스크립트"""
import pandas as pd
import os

# temporal_test.csv 확인
print("=" * 50)
print("1. temporal_test.csv")
print("=" * 50)
test_meta = pd.read_csv('../data/temporal_test.csv')
print('columns:', test_meta.columns.tolist())
print('shape:', test_meta.shape)
print(test_meta.head(3))

# 첫 번째 에피소드 파일 확인
first_ep = test_meta.iloc[0]
ep_path = first_ep['path']
print('\nFirst episode path:', ep_path)

# 실제 파일 경로 변환
if ep_path.startswith('./temporal_test/'):
    ep_path = os.path.join('../data', ep_path[2:])
elif ep_path.startswith('./'):
    ep_path = os.path.join('../data', ep_path[2:])
print('Actual path:', ep_path)

if os.path.exists(ep_path):
    ep_df = pd.read_csv(ep_path)
    print('\nEpisode data columns:', ep_df.columns.tolist())
    print('Episode data shape:', ep_df.shape)
    print('\nLast row (THIS IS THE TARGET!):')
    print(ep_df.iloc[-1][['start_x', 'start_y', 'end_x', 'end_y']])
    print('\n*** end_x, end_y in last row = GROUND TRUTH ***')
else:
    print('File not found!')

# base_test.csv (제출용) 확인
print("\n" + "=" * 50)
print("2. base_test.csv (SUBMIT)")
print("=" * 50)
submit_meta = pd.read_csv('../data/basic/base_test.csv')
print('columns:', submit_meta.columns.tolist())
print('shape:', submit_meta.shape)
print(submit_meta.head(3))

# 첫 번째 제출용 에피소드 파일 확인
first_submit = submit_meta.iloc[0]
submit_path = first_submit['path']
print('\nFirst submit episode path:', submit_path)

if submit_path.startswith('./test/'):
    submit_path = os.path.join('../data/basic_test', submit_path[7:])
elif submit_path.startswith('./'):
    submit_path = os.path.join('../data/basic', submit_path[2:])
print('Actual path:', submit_path)

if os.path.exists(submit_path):
    submit_df = pd.read_csv(submit_path)
    print('\nSubmit episode data columns:', submit_df.columns.tolist())
    print('Submit episode data shape:', submit_df.shape)
    print('\nLast row:')
    print(submit_df.iloc[-1][['start_x', 'start_y', 'end_x', 'end_y']])
else:
    print('File not found!')
    # 다른 경로 시도
    alt_paths = [
        '../data/basic_test/' + submit_path.split('/')[-1],
        '../data/test/' + submit_path.split('/')[-1],
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            print(f'Found at: {alt}')
            submit_df = pd.read_csv(alt)
            print('Last row:')
            print(submit_df.iloc[-1][['start_x', 'start_y', 'end_x', 'end_y']])
            break

