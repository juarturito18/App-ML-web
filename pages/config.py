import streamlit as st

def set_page(
    title: str,
    icon: str = "📊",
    remove_top_padding: bool = True
):
    # 1️⃣ Configuración base de la página
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 2️⃣ CSS global (opcional)
    if remove_top_padding:
        st.markdown(
            """
            <style>
            /* Quitar padding superior */
            .block-container {
                padding-top: 2rem;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

def section(title, content, bg, border):
    st.markdown(
        f"""
        <div style="
            background-color: {bg};
            padding: 1.2rem;
            border-radius: 10px;
            border-left: 6px solid {border};
            margin-bottom: 1.5rem;
        ">
            <h3 style="margin-top:0; font-weight: bold;">{title}</h3>
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )