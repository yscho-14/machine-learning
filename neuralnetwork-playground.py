import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler

# Matplotlib 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

@st.cache_data # 데이터셋 생성 캐싱
def generate_data(dataset_name, noise):
    """선택된 데이터셋을 생성합니다."""
    if dataset_name == '달 모양':
        X, y = make_moons(n_samples=200, noise=noise, random_state=0)
    elif dataset_name == '원형':
        X, y = make_circles(n_samples=200, noise=noise, factor=0.5, random_state=1)
    elif dataset_name == '선형 분리':
        X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=0, cluster_std=1.0 + noise * 5)
    
    # 데이터 스케일링
    X = StandardScaler().fit_transform(X)
    return X, y

def plot_decision_boundary(X, y, model, ax):
    """모델의 결정 경계를 시각화합니다."""
    # 1. 메쉬그리드 생성
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 2. 모델 예측
    # ravel()은 다차원 배열을 1차원으로 만듦
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # 3. 결정 경계 및 데이터 포인트 플로팅
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    return scatter


# -------------------------------------------------------------------
# Streamlit App UI
# -------------------------------------------------------------------

st.set_page_config(page_title="Streamlit MLP Playground", layout="wide")

st.title("🧠 Streamlit으로 만드는 신경망 플레이그라운드")
st.write("좌측 사이드바에서 데이터셋과 모델 하이퍼파라미터를 조정하고 **'모델 훈련'** 버튼을 눌러 결과를 확인하세요.")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 설정")

# 1. 데이터셋 선택
dataset_name = st.sidebar.selectbox(
    '1. 데이터셋 선택',
    ('달 모양', '원형', '선형 분리')
)

# 2. 노이즈 조절
noise = st.sidebar.slider('2. 노이즈 추가', 0.0, 0.5, 0.1, 0.05)

# 3. 모델 하이퍼파라미터
st.sidebar.subheader("🧠 신경망 모델 설정")

hidden_layer_sizes_str = st.sidebar.text_input('3. 은닉층(Hidden Layers) 구조', '10,5')
st.sidebar.caption("예: 10 (뉴런 10개인 층 1개), 10,5 (첫 층 10개, 둘째 층 5개)")

# 문자열을 정수 튜플로 변환
try:
    hidden_layer_sizes = tuple(map(int, hidden_layer_sizes_str.split(',')))
except:
    st.sidebar.error("쉼표로 구분된 숫자를 입력해주세요 (예: 10, 5)")
    st.stop()


activation = st.sidebar.selectbox(
    '4. 활성화 함수 (Activation)',
    ('relu', 'tanh', 'logistic', 'identity')
)

solver = st.sidebar.selectbox(
    '5. 최적화 알고리즘 (Solver)',
    ('adam', 'sgd', 'lbfgs')
)

learning_rate_init = st.sidebar.select_slider(
    '6. 학습률 (Learning Rate)',
    options=[0.0001, 0.001, 0.01, 0.1, 1.0],
    value=0.01
)

max_iter = st.sidebar.slider('7. 최대 반복 횟수 (Epochs)', 10, 2000, 500, 10)


# --- 메인 화면 ---
col1, col2 = st.columns([2, 1])

# 데이터 생성
X, y = generate_data(dataset_name, noise)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

with col1:
    st.subheader("📊 데이터 분포 및 결정 경계")
    
    # 훈련 버튼을 누르기 전, 원본 데이터만 표시
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu, edgecolors='k')
    ax.set_title("원본 데이터 분포")
    ax.set_xticks(())
    ax.set_yticks(())
    
    # 플레이스홀더를 사용해 그래프 영역을 미리 잡아둠
    plot_placeholder = st.pyplot(fig)

# 훈련 시작 버튼
if st.sidebar.button('🚀 모델 훈련', use_container_width=True):
    # 1. 모델 생성
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=0
    )

    # 2. 모델 훈련
    with st.spinner('모델이 훈련중입니다...'):
        model.fit(X_train, y_train)

    # 3. 성능 평가
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # 4. 결과 시각화
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_decision_boundary(X_train, y_train, model, ax)
    ax.set_title(f"모델 결정 경계 (테스트 정확도: {test_score:.2f})")
    
    # 플레이스홀더에 새로운 그래프를 업데이트
    plot_placeholder.pyplot(fig)
    
    # 메인 화면에 결과 출력
    with col2:
        st.subheader("📈 훈련 결과")
        st.metric(label="훈련 데이터 정확도", value=f"{train_score:.4f}")
        st.metric(label="테스트 데이터 정확도", value=f"{test_score:.4f}")
        
        st.write("---")
        st.write("최종 손실 (Loss):", f"{model.loss_:.4f}")
        st.write("반복 횟수:", f"{model.n_iter_}")


with col2:
    st.subheader("💡 사용 방법")
    st.info("""
    1.  **데이터셋**과 **노이즈**를 선택하여 문제의 난이도를 조절하세요.
    2.  **은닉층 구조**, **활성화 함수** 등 신경망 모델의 주요 파라미터를 변경해보세요.
    3.  **'모델 훈련'** 버튼을 눌러 왼쪽 그래프에 나타나는 **결정 경계(Decision Boundary)**의 변화를 관찰하세요.
    4.  결정 경계는 모델이 데이터를 어떻게 두 그룹으로 나누고 있는지 보여주는 선입니다.
    """)
