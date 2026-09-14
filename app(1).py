import io
import os
import re
from typing import Optional, Tuple

import anthropic
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="데이터로 감염병 유행을 추적하라!",
    page_icon="🦠",
    layout="wide",
)

st.title("🦠 데이터로 감염병 유행을 추적하라!")
st.caption("통합과학2 · 과학과 미래사회 | 실제 감염병 데이터 기반 탐구 도우미")

st.info(
    "이 앱은 AI가 정답을 대신 써 주는 도구가 아니라, "
    "데이터에서 사실을 찾고 해석의 한계를 점검하도록 돕는 탐구 도우미입니다."
)


# =========================================================
# 유틸리티 함수
# =========================================================
def read_csv_safely(uploaded_file) -> pd.DataFrame:
    """UTF-8 계열과 CP949를 순서대로 시도하여 CSV를 읽는다."""
    raw = uploaded_file.getvalue()
    last_error = None

    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as exc:
            last_error = exc

    raise ValueError(
        "CSV 파일을 읽지 못했습니다. UTF-8 또는 CP949 형식의 CSV인지 확인해주세요."
    ) from last_error


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """열 이름의 앞뒤 공백을 제거한다."""
    result = df.copy()
    result.columns = [str(c).strip() for c in result.columns]
    return result


def detect_year_and_value_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    '연도' 성격의 열과 '발생자 수' 성격의 수치 열을 최대한 안전하게 찾는다.
    찾지 못하면 숫자형 열을 후보로 사용한다.
    """
    year_col = None
    value_col = None

    # 1) 연도 열 탐색
    year_keywords = ("연도", "년도", "year")
    for col in df.columns:
        lower = str(col).lower()
        if any(k in lower for k in year_keywords):
            year_col = col
            break

    # 2) 발생자 수 열 탐색
    value_keywords = (
        "발생자수", "발생자 수", "환자수", "환자 수", "발생수",
        "신고수", "신고 수", "건수", "명", "cases", "case"
    )
    for col in df.columns:
        lower = str(col).lower().replace("_", " ")
        if any(k in lower for k in value_keywords):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= max(2, len(df) // 2):
                value_col = col
                break

    # 3) 그래도 못 찾으면 숫자형 열에서 추정
    numeric_candidates = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= max(2, len(df) // 2):
            numeric_candidates.append(col)

    if year_col is None:
        for col in numeric_candidates:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not vals.empty and vals.between(1900, 2100).mean() >= 0.8:
                year_col = col
                break

    if value_col is None:
        for col in numeric_candidates:
            if col != year_col:
                value_col = col
                break

    return year_col, value_col


def make_basic_chart(
    df: pd.DataFrame,
    year_col: str,
    value_col: str,
    log_scale: bool = False,
):
    """연도-발생자 수 기본 선그래프를 생성한다."""
    plot_df = df[[year_col, value_col]].copy()
    plot_df[year_col] = pd.to_numeric(plot_df[year_col], errors="coerce")
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna().sort_values(year_col)

    if plot_df.empty:
        raise ValueError("그래프를 그릴 수 있는 숫자형 데이터가 없습니다.")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        plot_df[year_col],
        plot_df[value_col],
        marker="o",
        linewidth=2,
    )
    ax.set_title("연도별 감염병 발생자 수 변화")
    ax.set_xlabel("연도")
    ax.set_ylabel("발생자 수(명)")
    ax.grid(True, alpha=0.25)

    if log_scale:
        positive = plot_df[value_col] > 0
        if positive.all():
            ax.set_yscale("log")
            ax.set_ylabel("발생자 수(명, 로그 척도)")
        else:
            st.warning("0 이하 값이 있어 로그 스케일을 적용하지 않았습니다.")

    # 연도는 가능한 경우 정수 눈금으로 표시
    try:
        years = plot_df[year_col].astype(int).tolist()
        if len(years) <= 20:
            ax.set_xticks(years)
            ax.tick_params(axis="x", rotation=45)
    except Exception:
        pass

    fig.tight_layout()
    return fig


def dataframe_for_prompt(df: pd.DataFrame, max_rows: int = 200) -> str:
    """
    API 비용과 입력 길이를 제한하면서도 데이터 근거를 충분히 전달한다.
    200행 이하는 전체, 초과 시 앞/뒤 일부와 요약을 제공한다.
    """
    if len(df) <= max_rows:
        return df.to_csv(index=False)

    head_n = max_rows // 2
    tail_n = max_rows - head_n
    head = df.head(head_n).to_csv(index=False)
    tail = df.tail(tail_n).to_csv(index=False)

    numeric_summary = df.describe(include="all").transpose().to_string()
    return (
        f"[전체 행 수: {len(df)}]\n"
        f"[앞 {head_n}행]\n{head}\n"
        f"[뒤 {tail_n}행]\n{tail}\n"
        f"[요약 통계]\n{numeric_summary}"
    )


def get_secret_api_key() -> str:
    """Streamlit Secrets 또는 환경변수에서 API 키를 읽는다."""
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return str(st.secrets["ANTHROPIC_API_KEY"]).strip()
    except Exception:
        pass

    return os.getenv("ANTHROPIC_API_KEY", "").strip()


def build_prompt(
    df: pd.DataFrame,
    question: str,
    prediction: str,
) -> str:
    data_text = dataframe_for_prompt(df)

    return f"""
너는 고등학교 1학년 통합과학2 '과학과 미래사회' 수업의 데이터 탐구 코치이다.
학생이 실제 감염병 발생 데이터를 근거로 스스로 결론을 구성하도록 도와라.

[절대 지켜야 할 규칙]
1. 아래 CSV에 들어 있는 정보만 '데이터에서 확인된 사실'로 말한다.
2. CSV에 없는 원인, 정책, 의학적 배경, 유행 원인 등은 사실처럼 단정하지 않는다.
3. 외부 지식이 필요하면 반드시 '추가 확인이 필요한 가설' 또는 '추가 자료가 필요함'으로 표시한다.
4. 상관관계와 인과관계를 구분한다.
5. 숫자를 언급할 때는 가능하면 실제 연도와 수치를 함께 제시한다.
6. 학생의 '나의 예상'을 존중하되, 데이터와 맞지 않으면 근거를 들어 수정하도록 안내한다.
7. 고등학교 1학년이 이해할 수 있는 간결한 과학 언어를 사용한다.
8. 답을 대신 써 주기보다 학생이 근거를 찾아 설명할 수 있게 돕는다.
9. 존재하지 않는 열, 값, 통계치를 만들어내지 않는다.

[학생의 질문]
{question}

[학생의 분석 전 예상]
{prediction if prediction.strip() else "입력하지 않음"}

[업로드 데이터]
{data_text}

[응답 형식]
반드시 아래 6개 제목을 정확히 사용해 순서대로 작성하라.

### ① 데이터에서 직접 확인되는 사실
- 데이터에 실제로 있는 수치와 연도를 근거로 2~4개 제시

### ② 추천 시각화
- 적절한 그래프 1~2개와 그 이유
- 현재 데이터 구조에서 가능한 시각화만 제안

### ③ 발견할 수 있는 패턴
- 증가·감소·급격한 변화·이상치 등
- 과장 없이 데이터에 보이는 패턴만 설명

### ④ 가능한 해석 또는 가설
- 원인으로 확정하지 말고 '가능성' 또는 '가설'로 표현
- 검증에 필요한 자료를 함께 언급

### ⑤ 이 데이터의 한계
- 이 CSV만으로는 알 수 없는 정보와 성급한 결론의 위험을 설명

### ⑥ 다음 탐구를 위해 필요한 데이터와 탐구 질문
- 추가하면 좋은 데이터 2~4개
- 이어서 탐구할 질문 2개

마지막 줄에는 반드시 다음 문장을 그대로 넣어라.
데이터가 보여 주는 사실과 그 원인에 대한 해석은 구분해야 합니다.
""".strip()


def call_claude(api_key: str, model_name: str, prompt: str) -> str:
    """Anthropic Messages API 호출."""
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model_name,
        max_tokens=1800,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    # 텍스트 블록만 안전하게 이어 붙임
    parts = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)

    result = "\n".join(parts).strip()
    if not result:
        raise RuntimeError("AI 응답에서 텍스트를 찾지 못했습니다.")
    return result


# =========================================================
# 사이드바
# =========================================================
st.sidebar.header("⚙️ 설정")

secret_key = get_secret_api_key()
if secret_key:
    st.sidebar.success("교사용 비밀 API 키가 설정되어 있습니다.")
    api_key = secret_key
else:
    api_key = st.sidebar.text_input(
        "Claude API 키",
        type="password",
        help="연수·개인 테스트용입니다. 학생 공개 앱에서는 Streamlit Secrets 사용을 권장합니다.",
    )

model_name = st.sidebar.text_input(
    "Claude 모델명",
    value="claude-sonnet-4-6",
    help="연수 자료의 기본값입니다. 계정에서 사용 가능한 모델명이 다르면 수정하세요.",
)

uploaded = st.sidebar.file_uploader(
    "CSV 파일 업로드",
    type=["csv"],
    help="UTF-8 또는 CP949 형식의 CSV를 지원합니다.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "🔐 실제 수업 배포 시 API 키를 코드나 GitHub에 넣지 마세요. "
    "Streamlit Community Cloud의 Secrets에 저장하세요."
)


# =========================================================
# 메인 화면
# =========================================================
if uploaded is None:
    st.subheader("1. 탐구 데이터 준비")
    st.write(
        "왼쪽 사이드바에서 CSV 파일을 업로드하세요. "
        "예: `백일해_2015-2024_전국발생자수.csv`"
    )
    st.stop()

try:
    df = normalize_columns(read_csv_safely(uploaded))
except Exception as exc:
    st.error(f"CSV 파일을 읽는 중 문제가 발생했습니다: {exc}")
    st.info(
        "확인할 점: 파일 확장자가 CSV인지, 첫 행에 열 이름이 있는지, "
        "UTF-8 또는 CP949 인코딩인지 확인해주세요."
    )
    st.stop()

if df.empty:
    st.error("업로드한 CSV에 데이터 행이 없습니다.")
    st.stop()

# 데이터 기본 정보
st.subheader("1. 데이터 확인")
c1, c2, c3 = st.columns(3)
c1.metric("행 수", f"{len(df):,}")
c2.metric("열 수", f"{len(df.columns):,}")
c3.metric("결측값 수", f"{int(df.isna().sum().sum()):,}")

with st.expander("📋 데이터 미리보기", expanded=True):
    st.dataframe(df, use_container_width=True, hide_index=True)

numeric_df = df.apply(pd.to_numeric, errors="coerce")
numeric_cols = [c for c in numeric_df.columns if numeric_df[c].notna().sum() > 0]

with st.expander("📊 숫자형 열 기초 통계"):
    if numeric_cols:
        st.dataframe(
            numeric_df[numeric_cols].describe().transpose(),
            use_container_width=True,
        )
    else:
        st.info("기초 통계를 계산할 수 있는 숫자형 열이 없습니다.")


# 기본 그래프
st.subheader("2. 기본 그래프 살펴보기")

year_col, value_col = detect_year_and_value_columns(df)

if year_col and value_col:
    st.caption(f"자동 인식: x축 = `{year_col}` / y축 = `{value_col}`")
    log_scale = st.checkbox(
        "로그 스케일로 보기",
        value=False,
        help="2024년처럼 매우 큰 값 때문에 이전 연도 변화가 잘 안 보일 때 사용해보세요.",
    )

    if st.button("📈 기본 그래프 보기", type="secondary"):
        try:
            fig = make_basic_chart(df, year_col, value_col, log_scale)
            st.pyplot(fig, clear_figure=True)
        except Exception as exc:
            st.error(f"그래프 생성 중 문제가 발생했습니다: {exc}")
else:
    st.warning(
        "연도 열과 발생자 수 열을 자동으로 찾지 못했습니다. "
        "AI 분석은 가능하지만 기본 선그래프는 생성하지 않습니다."
    )


# 학생 사고 선행
st.subheader("3. AI 분석 전에 먼저 생각하기")

prediction = st.text_area(
    "🧠 나의 예상",
    placeholder=(
        "예: 2024년에 발생자 수가 이전 연도보다 크게 증가했을 것 같다. "
        "하지만 이 데이터만으로 증가 원인은 알 수 없을 것 같다."
    ),
    height=110,
)

question = st.text_area(
    "❓ 데이터에 대해 궁금한 점",
    placeholder=(
        "예: 2024년 값은 이전 연도와 비교해 어떤 특징이 있나요?\n"
        "또는: 이 데이터만으로 2024년 증가 원인을 알 수 있나요?"
    ),
    height=120,
)

with st.expander("💡 탐구 질문 예시"):
    st.markdown(
        """
- 발생자 수가 가장 크게 증가한 시기는 언제인가?
- 전체적인 변화 추세는 어떠한가?
- 2024년 값은 이전 연도와 비교해 어떤 특징이 있는가?
- 이 데이터만으로 2024년 증가 원인을 알 수 있는가?
- 원인을 확인하려면 어떤 데이터가 더 필요한가?
        """
    )

analyze = st.button("🤖 AI 탐구 시작", type="primary", use_container_width=True)

if analyze:
    if not question.strip():
        st.warning("먼저 데이터에 대해 궁금한 점을 입력해주세요.")
    elif not api_key:
        st.warning(
            "Claude API 키가 없습니다. 사이드바에 키를 입력하거나 "
            "Streamlit Secrets에 ANTHROPIC_API_KEY를 설정해주세요."
        )
    elif not model_name.strip():
        st.warning("사용할 Claude 모델명을 입력해주세요.")
    else:
        try:
            prompt = build_prompt(df, question, prediction)

            with st.spinner("AI가 데이터를 근거로 탐구 방향을 정리하고 있습니다..."):
                result = call_claude(api_key, model_name.strip(), prompt)

            st.subheader("4. AI 탐구 코치의 분석")
            st.markdown(result)

            st.session_state["last_ai_result"] = result
            st.session_state["last_question"] = question
            st.session_state["last_prediction"] = prediction

        except anthropic.AuthenticationError:
            st.error(
                "API 인증에 실패했습니다. API 키가 올바른지 확인해주세요."
            )
        except anthropic.RateLimitError:
            st.error(
                "API 사용 한도 또는 호출 빈도를 초과했습니다. 잠시 후 다시 시도하거나 사용량을 확인해주세요."
            )
        except anthropic.BadRequestError as exc:
            st.error(
                "API 요청 형식 또는 모델 설정에 문제가 있습니다. "
                "사이드바의 모델명이 현재 계정에서 사용 가능한지 확인해주세요."
            )
            st.caption(str(exc))
        except anthropic.APIConnectionError:
            st.error(
                "Anthropic 서버에 연결하지 못했습니다. 네트워크 상태를 확인한 뒤 다시 시도해주세요."
            )
        except anthropic.APIStatusError as exc:
            st.error(f"Anthropic API 오류가 발생했습니다. 상태 코드: {exc.status_code}")
            st.caption(str(exc))
        except Exception as exc:
            st.error("AI 분석 중 예상하지 못한 오류가 발생했습니다.")
            st.exception(exc)


# 최종 해석
st.subheader("5. 나의 최종 해석")
final_interpretation = st.text_area(
    "✍️ AI의 도움을 받은 뒤, 데이터 근거를 사용해 내가 다시 설명해보세요.",
    placeholder=(
        "예: 2024년 발생자 수는 이전 연도에 비해 매우 크게 증가했다. "
        "그러나 이 자료에는 연령, 지역, 검사 건수, 예방접종률 등이 없으므로 "
        "증가 원인을 이 데이터만으로 단정할 수 없다."
    ),
    height=150,
)

st.caption(
    "✅ 좋은 데이터 탐구는 '관찰된 사실'과 '원인에 대한 해석'을 구분합니다."
)

# 결과 다운로드(학생 기록용)
if "last_ai_result" in st.session_state:
    record_text = f"""[탐구 질문]
{st.session_state.get("last_question", "")}

[AI 분석 전 나의 예상]
{st.session_state.get("last_prediction", "")}

[AI 탐구 코치 분석]
{st.session_state.get("last_ai_result", "")}

[나의 최종 해석]
{final_interpretation}
"""
    st.download_button(
        "💾 탐구 기록 TXT로 저장",
        data=record_text.encode("utf-8-sig"),
        file_name="감염병_데이터_탐구기록.txt",
        mime="text/plain",
        use_container_width=True,
    )
