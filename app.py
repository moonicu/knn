import streamlit as st
import joblib
import pandas as pd
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

# 모드별 리소스 지정
if mode_key == 'pre6':
    model_save_dir = 'saved_models_pre6'
    thresholds_file = 'thresholds_pre6.csv'
    x_columns = ['bwei', 'gad', 'mage', 'gran', 'chor', 'sterp']  # 학습 시 컬럼 순서 준수
elif mode_key == 'pre':
    model_save_dir = 'saved_models_pre'
    thresholds_file = 'thresholds_pre.csv'
    x_columns = ['mage', 'gran', 'parn', 'amni', 'mulg', 'bir', 'prep', 'dm', 'htn', 'chor',
                 'prom', 'ster', 'sterp', 'sterd', 'atbyn', 'delm', 'gad', 'sex', 'bwei']
else:  # 'post'
    model_save_dir = 'saved_models_post'
    thresholds_file = 'thresholds_post.csv'
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
# 성별(1,2,3)과 체중은 일부 모드에서 x_columns에 없을 수 있으므로 조건부 표기
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

# 맵 정의(라벨/옵션)
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
# Threshold 불러오기
# ======================
try:
    threshold_df = pd.read_csv(thresholds_file)
    # 기대 컬럼: target, model, threshold
    thresh_map = threshold_df.set_index(['target', 'model'])['threshold'].to_dict()
except Exception as e:
    st.error(t(
        f"임계값 파일을 불러오지 못했습니다: {thresholds_file}\n{e}",
        f"Failed to load thresholds file: {thresholds_file}\n{e}",
        lang
    ))
    st.stop()

# ======================
# 모델 로드 (모드별 캐시)
# ======================
@st.cache_resource(show_spinner=False)
def load_best_models(model_dir: str, y_cols: list):
    best_models = {}
    for y_col in y_cols:
        for model_name in ['LightGBM', 'XGBoost', 'RandomForest']:
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
run_btn = st.button(t("예측 실행", "Run Prediction", lang))

if run_btn:
    if not models:
        st.warning(t(
            f"모델을 찾지 못했습니다. 폴더를 확인하세요: {model_save_dir}",
            f"No models found. Please check folder: {model_save_dir}",
            lang
        ))
    else:
        predictions = {}
        used_model = {}
        used_thresh = {}

        for y_col in ALL_Y_COLUMNS:
            for model_name in ['LightGBM', 'XGBoost', 'RandomForest']:
                key = (y_col, model_name)
                if key in models:
                    model = models[key]
                    try:
                        prob = model.predict_proba(new_X_data)[0, 1]
                    except Exception:
                        # 확률 예측 불가 모델은 스킵
                        continue
                    thr = thresh_map.get((y_col, model_name), 0.5)
                    mark = "★" if prob >= thr else ""
                    predictions[y_col] = {
                        t('확률(%)', 'Probability (%)', lang): f"{prob*100:.2f}%",
                        t('플래그', 'Flag', lang): mark,
                        t('모델', 'Model', lang): model_name,
                        t('임계값', 'Threshold', lang): f"{thr:.3f}"
                    }
                    used_model[y_col] = model_name
                    used_thresh[y_col] = thr
                    break  # 우선순위(LGBM→XGB→RF) 중 처음 성공한 모델 사용

        # 데이터프레임 구성
        comp_targets = [y for y in ALL_Y_COLUMNS if y not in RESUS_TARGETS]

        resus_df = pd.DataFrame.from_dict({k: v for k, v in predictions.items() if k in RESUS_TARGETS}, orient='index')
        comp_df = pd.DataFrame.from_dict({k: v for k, v in predictions.items() if k in comp_targets}, orient='index')

        # 표시 이름 컬럼 추가
        if not resus_df.empty:
            resus_df.insert(0, t('항목', 'Outcome', lang), [y_display_names.get(k, k) for k in resus_df.index])
        if not comp_df.empty:
            comp_df.insert(0, t('항목', 'Outcome', lang), [y_display_names.get(k, k) for k in comp_df.index])

        # 출력
        st.subheader(t("* 신생아 소생술 관련 예측", "* Resuscitation Predictions", lang))
        if resus_df.empty:
            st.info(t("표시할 예측 결과가 없습니다.", "No predictions to display.", lang))
        else:
            st.dataframe(resus_df.reset_index(drop=True), use_container_width=True)

        st.subheader(t("* 미숙아 합병증 및 예후 예측", "* Complication Predictions", lang))
        if comp_df.empty:
            st.info(t("표시할 예측 결과가 없습니다.", "No predictions to display.", lang))
        else:
            st.dataframe(comp_df.reset_index(drop=True), use_container_width=True)

        # ======================
        # 결과 TXT 다운로드
        # ======================
        if patient_id:
            output = io.StringIO()
            output.write(f"Patient ID: {patient_id}\nDate: {datetime.today().strftime('%Y-%m-%d')}\n")
            output.write(f"Mode: {selected_label}\nModel dir: {model_save_dir}\nThreshold file: {thresholds_file}\n\n")


            # 입력 정보
            output.write("[입력 정보 / Input Information]\n")
            output.write(f"gaw: {gaw}\n")
            output.write(f"gawd: {gawd}\n")
            output.write(f"gad: {gad}\n")
            # x_columns 순서대로 기록
            for col in x_columns:
                if col in inputs:
                    output.write(f"{col}: {inputs[col]}\n")
            output.write("\n[예측 결과 / Prediction Results]\n")

            # 표형식 문자열
            if not resus_df.empty:
                output.write("[Resuscitation Predictions]\n")
                output.write(resus_df.to_string(index=False))
                output.write("\n\n")
            if not comp_df.empty:
                output.write("[Complication Predictions]\n")
                output.write(comp_df.to_string(index=False))
                output.write("\n")

            st.download_button(
                label=t("결과 TXT 다운로드", "Download Results TXT", lang),
                data=output.getvalue(),
                file_name=f"{patient_id}_result.txt",
                mime="text/plain"
            )
