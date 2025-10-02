import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime

st.set_page_config(page_title="KNN Neonatal Outcome Predictor", layout="wide")

# ======================
# 언어 / 이름 매핑
# ======================
def t(ko, en, lang):
    return ko if lang == '한국어' else en

# 그룹 구분(공통)
ALL_Y_COLUMNS = ['resu', 'resuo', 'resup', 'resui', 'resuh', 'resue', 'resuc', 'rds', 'sft', 'sftw',
                 'als', 'mph', 'ph', 'bpdyn', 'bpdm', 'pdad', 'acl', 'lbp', 'ivh2', 'ivh3', 'pvl', 'seps',
                 'ntet', 'pmio', 'eythtran', 'deathyn','supyn']
RESUS_TARGETS = ['resu', 'resuo', 'resup', 'resui', 'resuh', 'resue', 'resuc', 'sft', 'sftw']

y_display_ko = {
    'resu': '초기 소생술 필요 유무', 'resuo': '초기 소생술 산소', 'resup': '초기 소생술 양압 환기', 'resui': '초기 소생술 기도 삽관',
    'resuh': '초기 소생술 심장마사지', 'resue': '초기 소생술 Epinephrine', 'resuc': '초기 소생술 CPAP',
    'rds': '신생아 호흡곤란증후군', 'sft': '폐표면활성제 사용', 'sftw': '폐활성제 출생 즉시 사용',
    'als': '공기누출증후군', 'mph': '대량 폐출혈', 'ph': '폐동맥 고혈압', 'bpdyn': '기관지폐이형성증 여부(≥mild BPD)',
    'bpdm': '중등증 기관지폐이형성증(≥moderate BPD)', 'pdad': 'PDA 약물 치료', 'acl': '동맥관 결찰술', 'lbp': '저혈압',
    'ivh2': '뇌실내출혈 (Grade≥2)', 'ivh3': '중증 뇌실내출혈 (Grade≥3)', 'pvl': '백질연화증', 'seps': '패혈증',
    'ntet': '괴사성 장염', 'pmio': '망막증 수술', 'eythtran': '적혈구 수혈', 'deathyn': 'NICU 입원중 사망', 'supyn': '퇴원시 보조 장비 필요'
}
y_display_en = {
    'resu': 'Resuscitation needed', 'resuo': 'Oxygen', 'resup': 'PPV', 'resui': 'Intubation',
    'resuh': 'Chest compression', 'resue': 'Epinephrine', 'resuc': 'CPAP',
    'rds': 'RDS', 'sft': 'Surfactant use', 'sftw': 'Immediate surfactant',
    'als': 'Air leak', 'mph': 'Massive pulmonary hemorrhage', 'ph': 'Pulmonary hypertension',
    'bpdyn': '≥ Mild BPD', 'bpdm': '≥ Moderate BPD', 'pdad': 'PDA medication', 'acl': 'PDA ligation', 'lbp': 'Hypotension',
    'ivh2': 'IVH (≥Grade 2)', 'ivh3': 'IVH (≥Grade 3)', 'pvl': 'PVL', 'seps': 'Sepsis',
    'ntet': 'NEC', 'pmio': 'ROP surgery', 'eythtran': 'RBC transfusion', 'deathyn': 'In-hospital death', 'supyn': 'Discharge support'
}

# ======================
# 사이드바: 언어/모드 (영-한 자동 변환)
# ======================
lang = st.sidebar.radio("언어 / Language", ['한국어', 'English'])

# 모드 라벨 사전: 내부키는 고정, 라벨은 언어별
MODE_OPTIONS = {
    'pre6': {'ko': '산전 단순 예측 (6개변수)', 'en': 'Prenatal – Simple (6 features)'},
    'pre':  {'ko': '산전 예측',               'en': 'Prenatal – Full'},
    'post': {'ko': '산후 예측',               'en': 'Postnatal – Full'},
}

mode_label = t("예측 모드 / Prediction Mode", "Prediction Mode", lang)

# 현재 언어에 맞는 라벨 리스트와 역매핑 구성
if lang == '한국어':
    display_labels = [MODE_OPTIONS[k]['ko'] for k in ['pre6', 'pre', 'post']]
else:
    display_labels = [MODE_OPTIONS[k]['en'] for k in ['pre6', 'pre', 'post']]

label2key = {MODE_OPTIONS['pre6']['ko']: 'pre6', MODE_OPTIONS['pre6']['en']: 'pre6',
             MODE_OPTIONS['pre']['ko']:  'pre',  MODE_OPTIONS['pre']['en']:  'pre',
             MODE_OPTIONS['post']['ko']: 'post', MODE_OPTIONS['post']['en']: 'post'}

selected_label = st.sidebar.radio(mode_label, display_labels, index=0)  # default: pre6
mode_key = label2key[selected_label]  # 내부 사용 키

# 모드별 리소스 지정 (metrics_* 파일 사용)
if mode_key == 'pre6':
    model_save_dir = 'saved_models_pre6'
    metrics_file = 'saved_models_pre6/model_performance_pre6.csv'
    x_columns = ['bwei', 'gad', 'mage', 'gran', 'chor', 'sterp']  # 학습 시 컬럼 순서 준수
elif mode_key == 'pre':
    model_save_dir = 'saved_models_pre'
    metrics_file = 'saved_models_pre6/model_performance_pre.csv'
    x_columns = ['mage', 'gran', 'parn', 'amni', 'mulg', 'bir', 'prep', 'dm', 'htn', 'chor',
                 'prom', 'ster', 'sterp', 'sterd', 'atbyn', 'delm', 'gad', 'sex', 'bwei']
else:  # 'post'
    model_save_dir = 'saved_models_post'
    metrics_file = 'saved_models_pre6/model_performance_post.csv'
    x_columns = ['mage', 'gran', 'parn', 'amni', 'mulg', 'bir', 'prep', 'dm', 'htn', 'chor',
                 'prom', 'ster', 'sterp', 'sterd', 'atbyn', 'delm', 'gad', 'sex', 'bwei']

# y 표시 이름
y_display_names = y_display_ko if lang == '한국어' else y_display_en

# ======================
# 제목/설명
# ======================
st.title("KNN Neonatal Outcome Predictor")
st.caption(t(
    "산전/산후 변수로 신생아 소생술 및 합병증 위험을 예측합니다.",
    "Predicts neonatal resuscitation and complications from prenatal/postnatal variables.",
    lang
))

# ======================
# 입력 위젯 유틸
# ======================
def get_selectbox(label_kr, label_en, options, labels_kr, labels_en, key=None):
    label = t(label_kr, label_en, lang)
    labels = labels_en if lang == 'English' else labels_kr
    return st.selectbox(label, options, format_func=lambda x: labels[options.index(x)], key=key)

# 공통(주수/일수 → gad)
col_gaw, col_gawd = st.columns(2)
with col_gaw:
    gaw = st.number_input(t("임신 주수", "Gestational Weeks", lang), 20, 50, 28)
with col_gawd:
    gawd = st.number_input(t("임신 일수", "Gestational Days", lang), 0, 6, 0)
gad = gaw * 7 + gawd

# 공통 입력(성별/체중) — 필요 시만 노출
show_sex = 'sex' in x_columns
show_bwei = 'bwei' in x_columns

if show_sex or show_bwei:
    c1, c2 = st.columns(2)
    if show_sex:
        with c1:
            sex = get_selectbox("성별", "Sex", [1, 2, 3], ["남아", "여아", "미분류"], ["Male", "Female", "Ambiguous"], key="sex")
    else:
        sex = None
    if show_bwei:
        with c2:
            bwei = st.number_input(t("출생 체중 (g)", "Birth Weight (g)", lang), 200, 5000, 1000, key="bwei")
    else:
        bwei = None
else:
    sex = None
    bwei = None

# 나머지 변수 위젯(모드에서 쓰는 변수만 렌더)
inputs = {}

def render_if_needed(var):
    if var not in x_columns:
        return
    if var == 'mage':
        inputs['mage'] = st.number_input(t("산모 나이", "Maternal Age", lang), 15, 99, 30, key='mage')
    elif var == 'gran':
        inputs['gran'] = st.number_input(t("임신력", "Gravidity", lang), 0, 10, 0, key='gran')
    elif var == 'parn':
        inputs['parn'] = st.number_input(t("출산력", "Parity", lang), 0, 10, 0, key='parn')
    elif var == 'amni':
        inputs['amni'] = get_selectbox("양수량", "Amniotic Fluid", [1, 2, 3, 4],
                                       ["정상", "과소", "과다", "모름"],
                                       ["Normal", "Oligo", "Poly", "Unknown"], key='amni')
    elif var == 'mulg':
        inputs['mulg'] = get_selectbox("다태 정보", "Multiplicity", [1, 2, 3, 4],
                                       ["Singleton", "Twin", "Triplet", "Quad 이상"],
                                       ["Singleton", "Twin", "Triplet", "Quad+"], key='mulg')
    elif var == 'bir':
        inputs['bir'] = get_selectbox("출생 순서", "Birth Order", [0, 1, 2, 3, 4],
                                      ["단태", "1st", "2nd", "3rd", "4th 이상"],
                                      ["Single", "1st", "2nd", "3rd", "4th+"], key='bir')
    elif var == 'prep':
        inputs['prep'] = get_selectbox("임신 과정", "Pregnancy Type", [1, 2],
                                       ["자연임신", "IVF"], ["Natural", "IVF"], key='prep')
    elif var == 'dm':
        inputs['dm'] = get_selectbox("당뇨", "Diabetes", [1, 2, 3],
                                     ["없음", "GDM", "Overt DM"], ["None", "GDM", "Overt"], key='dm')
    elif var == 'htn':
        inputs['htn'] = get_selectbox("고혈압", "Hypertension", [1, 2, 3],
                                      ["없음", "PIH", "Chronic HTN"], ["None", "PIH", "Chronic"], key='htn')
    elif var == 'chor':
        inputs['chor'] = get_selectbox("융모양막염", "Chorioamnionitis", [1, 2, 3],
                                       ["없음", "있음", "모름"], ["No", "Yes", "Unknown"], key='chor')
    elif var == 'prom':
        inputs['prom'] = get_selectbox("조기 양막 파열", "PROM", [1, 2, 3],
                                       ["없음", "있음", "모름"], ["No", "Yes", "Unknown"], key='prom')
    elif var == 'ster':
        inputs['ster'] = get_selectbox("스테로이드 사용", "Steroid Use", [1, 2, 3],
                                       ["없음", "있음", "모름"], ["No", "Yes", "Unknown"], key='ster')
    elif var == 'sterp':
        inputs['sterp'] = get_selectbox("스테로이드 완료 여부", "Steroid Completion", [0, 1, 2, 3],
                                        ["미투여", "미완료", "완료", "모름"],
                                        ["None", "Incomplete", "Complete", "Unknown"], key='sterp')
    elif var == 'sterd':
        inputs['sterd'] = get_selectbox("스테로이드 약제", "Steroid Type", [0, 1, 2, 3, 4],
                                        ["미투여", "Dexa", "Beta", "Dexa+Beta", "모름"],
                                        ["None", "Dexa", "Beta", "Dexa+Beta", "Unknown"], key='sterd')
    elif var == 'atbyn':
        inputs['atbyn'] = get_selectbox("항생제 사용", "Antibiotics", [1, 2],
                                        ["없음", "있음"], ["No", "Yes"], key='atbyn')
    elif var == 'delm':
        inputs['delm'] = get_selectbox("분만 방식", "Delivery Mode", [1, 2],
                                       ["질식분만", "제왕절개"], ["Vaginal", "Cesarean"], key='delm')
    elif var == 'gad':
        inputs['gad'] = gad
    elif var == 'sex':
        inputs['sex'] = sex if sex is not None else 1
    elif var == 'bwei':
        default_bwei = 1000 if bwei is None else bwei
        inputs['bwei'] = default_bwei

# 레이아웃 정리: 2열로 가볍게 배치
left_vars = ['mage','gran','parn','amni','mulg','bir','prep','dm','htn']
right_vars = ['chor','prom','ster','sterp','sterd','atbyn','delm']

lcol, rcol = st.columns(2)
with lcol:
    for v in left_vars:
        render_if_needed(v)
with rcol:
    for v in right_vars:
        render_if_needed(v)

# 필수(gad, sex, bwei)는 마지막에 보정
render_if_needed('gad')
render_if_needed('sex')
render_if_needed('bwei')

# x_columns 순서대로 DataFrame 구성
try:
    new_X_data = pd.DataFrame([[inputs[col] for col in x_columns]], columns=x_columns)
except KeyError as e:
    st.error(t(f"입력 누락: {e}", f"Missing input: {e}", lang))
    st.stop()

# 환자 식별자
patient_id = st.text_input(t("환자 등록번호 (저장시 파일명)", "Patient ID (for download)", lang), max_chars=20)

# ======================
# 모델 성능(Metrics) 불러오기 — 사용 파일 열 구조에 맞춘 전용 로더
# ======================
import re
import numpy as np

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _std_model_name(x: str) -> str:
    if not isinstance(x, str):
        return str(x)
    s = x.strip().lower()
    if s in {"lightgbm", "lgbm", "lgb", "light-gbm"}:
        return "LightGBM"
    if s in {"xgboost", "xgb", "xg-boost"}:
        return "XGBoost"
    return x.strip()  # 그 외는 원문 유지

try:
    # 업로드하신 파일명을 그대로 사용 (pre6 모드)
    metrics_path = os.path.join(model_save_dir, 'model_performance_pre6.csv')

    # 콤마/탭 자동 감지
    metrics_df_raw = pd.read_csv(metrics_path, sep=None, engine="python")

    # 기대 컬럼 존재 확인 (대소문자/공백 허용)
    cols_norm = {re.sub(r'\s+', '', c).strip().lower(): c for c in metrics_df_raw.columns}
    need = {"target", "model", "f1_binary", "auc", "auprc"}
    if not need.issubset(set(cols_norm.keys())):
        missing = need - set(cols_norm.keys())
        raise ValueError(f"필수 컬럼 누락: {sorted(missing)}  (필요: {sorted(list(need))})")

    # 필요한 컬럼만 표준 키로 리네임
    metrics_df = metrics_df_raw.rename(columns={
        cols_norm["target"]: "target",
        cols_norm["model"]: "model",
        cols_norm["f1_binary"]: "f1",
        cols_norm["auc"]: "auc",
        cols_norm["auprc"]: "auprc",
    })

    # 모델명 표준화
    metrics_df["model"] = metrics_df["model"].apply(_std_model_name)

    # 숫자 캐스팅
    for m in ["f1", "auprc", "auc"]:
        metrics_df[m] = metrics_df[m].apply(_safe_float)

    # {(target, model) -> (f1, auprc, auc)}
    METRIC_MAP = {(str(r["target"]), str(r["model"])): (r["f1"], r["auprc"], r["auc"])
                  for _, r in metrics_df.iterrows()}

except Exception as e:
    st.warning(t(
        f"모델 성능 파일을 불러오지 못했습니다: {os.path.join(model_save_dir, 'model_performance_pre6.csv')}\n{e}",
        f"Failed to load metrics file: {os.path.join(model_save_dir, 'model_performance_pre6.csv')}\n{e}",
        lang
    ))
    METRIC_MAP = {}


# ======================
# 모델 로드 (LightGBM / XGBoost만)
# ======================
@st.cache_resource(show_spinner=False)
def load_best_models(model_dir: str, y_cols: list):
    best_models = {}
    for y_col in y_cols:
        for model_name in ['LightGBM', 'XGBoost']:  # RandomForest 제외
            path = os.path.join(model_dir, f"best_{model_name}_{y_col}.pkl")
            if os.path.exists(path):
                try:
                    best_models[(y_col, model_name)] = joblib.load(path)
                except Exception:
                    continue
    return best_models

models = load_best_models(model_save_dir, ALL_Y_COLUMNS)

# ======================
# 예측 실행
# ======================

def df_auto_height(n_rows: int, max_rows: int = None) -> int:
    """
    Streamlit dataframe 높이를 행 수에 맞춰 계산.
    기본 행 높이 ~38px, 헤더 ~38px, 약간의 패딩 포함.
    """
    if max_rows is not None:
        n_rows = min(n_rows, max_rows)
    row_px = 38
    header_px = 38
    padding_px = 16
    return header_px + n_rows * row_px + padding_px

def grade_level(f1, auprc, auc):
    # 등급 규칙
    if (f1 is not None and auprc is not None and auc is not None and
        not np.isnan([f1, auprc, auc]).any()):
        if (f1 >= 0.75) and (auprc >= 0.70) and (auc >= 0.80):
            return "🟢 High (임상 활용 후보)"
        elif (f1 >= 0.50) and (auprc >= 0.50) and (auc >= 0.75):
            return "🟡 Medium (스크리닝/참조)"
        else:
            return "🔴 Low (연구/참고용)"
    return "N/A"

def perf_string(f1, auprc, auc):
    # 소수 2자리 + 등급
    if (f1 is None) or (auprc is None) or (auc is None) or np.isnan([f1, auprc, auc]).any():
        core = "F1=N/A, AUPRC=N/A, AUC=N/A"
    else:
        core = f"F1={f1:.2f}, AUPRC={auprc:.2f}, AUC={auc:.2f}"
    return f"{core} {grade_level(f1, auprc, auc)}"

run_btn = st.button(t("예측 실행", "Run Prediction", lang))

if run_btn:
    if not models:
        st.warning(t(
            f"모델을 찾지 못했습니다. 폴더를 확인하세요: {model_save_dir}",
            f"No models found. Please check folder: {model_save_dir}",
            lang
        ))
    else:
        rows_resus = []
        rows_comp = []

        for y_col in ALL_Y_COLUMNS:
            outcome_name = y_display_names.get(y_col, y_col)

            row = {
                'Outcome': outcome_name,
                'XGBoost': "N/A",
                '모델성능(F1-score, AUPRC, AUC)': "N/A",
                'LightGBM': "N/A",
                '모델성능((F1-score, AUPRC, AUC))': "N/A"
            }

            # XGBoost
            key_xgb = (y_col, 'XGBoost')
            if key_xgb in models:
                try:
                    prob_xgb = models[key_xgb].predict_proba(new_X_data)[0, 1]
                    row['XGBoost'] = f"{prob_xgb*100:.2f}%"
                except Exception:
                    row['XGBoost'] = "N/A"
                f1, auprc, auc = METRIC_MAP.get((y_col, 'XGBoost'), (None, None, None))
                row['모델성능(F1-score, AUPRC, AUC)'] = perf_string(f1, auprc, auc)

            # LightGBM
            key_lgb = (y_col, 'LightGBM')
            if key_lgb in models:
                try:
                    prob_lgb = models[key_lgb].predict_proba(new_X_data)[0, 1]
                    row['LightGBM'] = f"{prob_lgb*100:.2f}%"
                except Exception:
                    row['LightGBM'] = "N/A"
                f1, auprc, auc = METRIC_MAP.get((y_col, 'LightGBM'), (None, None, None))
                row['모델성능((F1-score, AUPRC, AUC))'] = perf_string(f1, auprc, auc)

            # 그룹 분리
            if y_col in RESUS_TARGETS:
                rows_resus.append(row)
            else:
                rows_comp.append(row)

        # 데이터프레임 만들기
        resus_df = pd.DataFrame(rows_resus)
        comp_df = pd.DataFrame(rows_comp)

        # 출력
        st.subheader(t("* 신생아 소생술 관련 예측", "* Resuscitation Predictions", lang))
        if resus_df.empty:
            st.info(t("표시할 예측 결과가 없습니다.", "No predictions to display.", lang))
        else:
            st.dataframe(resus_df, use_container_width=True)

        st.subheader(t("* 미숙아 합병증 및 예후 예측", "* Complication Predictions", lang))
        if comp_df.empty:
            st.info(t("표시할 예측 결과가 없습니다.", "No predictions to display.", lang))
        else:
            # 17행까지는 스크롤 없이 한 화면에 보이도록 높이 지정
            height_comp = df_auto_height(len(comp_df), max_rows=17)
            st.dataframe(comp_df, use_container_width=True, height=height_comp)


        # ======================
        # 결과 TXT 다운로드 (XGBoost/LightGBM + 성능 함께 기록)
        # ======================
        if patient_id:
            output = io.StringIO()
            output.write(f"Patient ID: {patient_id}\nDate: {datetime.today().strftime('%Y-%m-%d')}\n")
            output.write(f"Mode: {selected_label}\nModel dir: {model_save_dir}\nMetrics file: {metrics_file}\n\n")

            # 입력 정보
            output.write("[입력 정보 / Input Information]\n")
            output.write(f"gaw: {gaw}\n")
            output.write(f"gawd: {gawd}\n")
            output.write(f"gad: {gad}\n")
            for col in x_columns:
                if col in inputs:
                    output.write(f"{col}: {inputs[col]}\n")

            def _write_block(title, df):
                if not df.empty:
                    output.write(f"\n[{title}]\n")
                    output.write(df.to_string(index=False))
                    output.write("\n")

            _write_block("Resuscitation Predictions", resus_df)
            _write_block("Complication Predictions", comp_df)

            st.download_button(
                label=t("결과 TXT 다운로드", "Download Results TXT", lang),
                data=output.getvalue(),
                file_name=f"{patient_id}_result.txt",
                mime="text/plain"
            )
