# 소유 판정 키트 (학습 → 대회 데이터 적용)

Sportec 공개 데이터 7경기로 "선수 집합 → H/A/D" 모델을 학습하고, 대회 ATD 에 적용해 제출 CSV 를 만든다.
대회 데이터로는 학습하지 않는다. 학습된 최종 모델(model_final.pt, params.json)이 같이 들어 있어서
**적용만 할 거면 6단계로 바로 가면 된다.**

## 폴더 구성
| 파일 | 역할 |
|---|---|
| check_sportec.py | Sportec XML 21개가 제대로 있는지 확인 (`--find <폴더>` 로 찾기) |
| prep_all.py | PathCRF 전처리: XML → 정답(소유 팀, dead) 붙은 표 |
| noise.py | 선수 잡음 흉내 (결측 구간, 좌표 오차, 중복) |
| build_ds.py | 5 fps 데이터셋. 공 결측 흉내 두 가지: BALL_STYLE=v3(기본, alive 짧은 결측 + dead 긴 결측), v1(기존) |
| models_new.py | 모델 부품 (Set Attention, PMA) |
| train.py | val(검증으로 선택) / test(한 번 채점) / final(7경기 전부로 최종 학습) |
| teamscore.py | 채점 (대회 채점 코드 사용) |
| run.py | 대회 ATD → 제출 CSV, `--score` 로 공개 5분 채점 |
| atd_ball_stats.py | 대회 ATD 의 공·선수 결측 통계 (잡음 흉내 설정을 실제 값에 맞출 때) |
| model_final.pt, params.json | 학습 끝난 최종 모델과 Viterbi 설정 |

## 0. 준비 (한 번만)
- PathCRF 저장소 (학습할 때만): `git clone https://github.com/hyunsungkim-ds/pathcrf.git` → 기본 위치 ~/pathcrf
- 대회 저장소가 ~/more-than-a-hack-2027 이 아니면 `export MTAH_REPO=<경로>`, PathCRF 가 ~/pathcrf 가 아니면 `export PATHCRF_DIR=<경로>`
- `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- `pip install kloppy rdp scipy scikit-learn pyarrow tqdm pandas numpy`

## 학습 (처음부터 다시 할 때)
1. Sportec XML 21개를 ~/pathcrf/data/sportec/{metadata,event,tracking}/ 에 두고 확인: `python check_sportec.py`
2. 전처리 (경기당 약 3분, 메모리 최대 약 4.5 GB): `python prep_all.py`
3. 데이터셋 두 벌 (경기마다):
   - `DS_DIR=ds python build_ds.py J03WMX`
   - `DS_DIR=ds_old BALL_STYLE=v1 python build_ds.py J03WMX`
   - 경기: J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9
4. 검증으로 설정 고르기 (2코어 CPU 약 30분): `DS_DIR=ds,ds_old python train.py val E4 20 45`
5. 테스트 채점 (한 번만): `DS_DIR=ds,ds_old python train.py test E4`
6-1. 7경기 전부로 최종 학습: `DS_DIR=ds,ds_old python train.py final E4` → model_final.pt, params.json

## 6. 대회 데이터에 적용
`python run.py --repo <대회 저장소> --data-dir <tactical 폴더가 든 데이터 폴더> --out submission.csv --score`
`python <대회 저장소>/evaluation/validate_submission.py submission.csv`

화면에는 진행 상황과 점수만 나온다. 공개 5분 점수는 확인용이고, 이걸 보고 설정을 고치지 않는다 (17개 구간으로는 비교 불가).

## (선택) 잡음 흉내를 실제 값에 맞추기
`python atd_ball_stats.py --repo <대회 저장소> --data-dir <데이터 폴더>`
나온 숫자로 build_ds.py 맨 위 BALL (alive_missing, dead_missing, alive_gap_s 등) 과 noise.py 의 LEVELS 를 바꾸고 3단계부터 다시.

## 결과 기록 (Sportec 테스트 경기 J03WR9, spellF1, dead 도 모델이 예측)
| 모델 | 기존 흉내 테스트 | 새 흉내 테스트 | 깨끗한 테스트 |
|---|---|---|---|
| E2 (기존 흉내만) | 0.681 | 0.656 | 0.746 |
| E3 (새 흉내만) | 0.581 | 0.699 | 0.719 |
| E4 (둘 다 섞음) = 최종 모델 설정 | 0.667 | 0.718 | 0.765 |
최종 모델은 테스트 경기까지 학습에 넣었으므로 따로 남은 채점 경기가 없다. 기대 성능은 E4 테스트 점수로 본다.
대회 공개 5분: E2 spellF1 0.537 (공식 기준선 ±50 은 같은 5분에서 0.326).

## 공유 메모
- 최종 모델: **E4** (`model_final.pt` + `params.json`). `run.py` 기본값이 이 파일들을 씀.
- 데이터는 저장소에 없음. 대회 데이터·Sportec 데이터는 각자 받아서 경로로 지정.
- 학습 시 외부 저장소 필요: `git clone https://github.com/hyunsungkim-ds/pathcrf.git ~/pathcrf`
- TODO: `prep_all.py` 가 읽는 `~/pathcrf/data/sportec_synced/*.parquet` 생성 방법 확인 후 기록.
