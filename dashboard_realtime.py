import streamlit as st
import socket
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import math 
import os 
from datetime import datetime

# --- IMPORTAÇÃO DA LÓGICA DO GRUPO (Leitura_sEMG_IMU.py) ---
try:
    from Leitura_sEMG_IMU import (
        processar_dados_marcha, 
        processar_dados_semg,
        gerar_relatorio_para_streamlit,
        gerar_grafico_evolucao, 
        REGRAS_SEMG,
        NOMES_PARAMETROS_SEMG
    )
except ImportError:
    st.error("Erro Crítico: O arquivo 'Leitura_sEMG_IMU.py' não foi encontrado. Certifique-se de que ele está na mesma pasta.")
    st.stop()


# =============================================================================
# 1. FUNÇÕES DE PERSISTÊNCIA E CONFIGURAÇÕES
# =============================================================================
ARQUIVO_PACIENTES = 'pacientes.csv'
ARQUIVO_SESSOES = 'dados_sessoes.csv' 
UDP_IP = "0.0.0.0"; 
UDP_PORT = 4210; 
DURACAO_COLETA = 15; 
COLUNAS_ESPERADAS = 7
LOGO_PATH = "neurostep_logo_escuro.png"

def carregar_pacientes():
    if os.path.exists(ARQUIVO_PACIENTES): return pd.read_csv(ARQUIVO_PACIENTES)
    else:
        df = pd.DataFrame(columns=['ID_PACIENTE', 'NOME', 'DATA_CADASTRO'])
        df.to_csv(ARQUIVO_PACIENTES, index=False)
        return df

def salvar_pacientes(df):
    df.to_csv(ARQUIVO_PACIENTES, index=False)

def salvar_dados_sessao(patient_id, met_marcha, params_semg, fs_real, duracao):
    if not os.path.exists(ARQUIVO_SESSOES):
        colunas = [
            'DATA_HORA', 'ID_PACIENTE', 'DURACAO_S', 'FS_REAL', 
            'CADENCIA', 'PCT_APOIO', 'ESTABILIDADE', 'MEDIA_APOIO', 'MEDIA_BALANCO',
            'RMS', 'MAV', 'LOG', 'WL', 'MNF', 'MDF'
        ]
        df = pd.DataFrame(columns=colunas)
        df.to_csv(ARQUIVO_SESSOES, index=False)
    
    df_sessoes = pd.read_csv(ARQUIVO_SESSOES)
    nova_sessao = {
        'DATA_HORA': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ID_PACIENTE': patient_id, 'DURACAO_S': duracao, 'FS_REAL': fs_real,
        'CADENCIA': met_marcha.get('cadencia', np.nan), 'PCT_APOIO': met_marcha.get('pct_apoio', np.nan),
        'ESTABILIDADE': met_marcha.get('estabilidade', np.nan), 'MEDIA_APOIO': met_marcha.get('media_apoio', np.nan),
        'MEDIA_BALANCO': met_marcha.get('media_balanco', np.nan),
        'RMS': params_semg.get('RMS', np.nan), 'MAV': params_semg.get('MAV', np.nan),
        'LOG': params_semg.get('LOG', np.nan), 'WL': params_semg.get('WL', np.nan),
        'MNF': params_semg.get('MNF', np.nan), 'MDF': params_semg.get('MDF', np.nan),
    }
    df_nova_sessao = pd.DataFrame([nova_sessao])
    df_atualizado = pd.concat([df_sessoes, df_nova_sessao], ignore_index=True)
    df_atualizado.to_csv(ARQUIVO_SESSOES, index=False)
    return True

def excluir_sessoes(lista_de_data_hora_original_a_excluir):
    """Lê todas as sessões e remove as linhas com base nos valores RAW da DATA_HORA."""
    if os.path.exists(ARQUIVO_SESSOES) and lista_de_data_hora_original_a_excluir:
        df_full = pd.read_csv(ARQUIVO_SESSOES)
        df_atualizado = df_full[~df_full['DATA_HORA'].isin(lista_de_data_hora_original_a_excluir)]
        df_atualizado.to_csv(ARQUIVO_SESSOES, index=False)
        return True
    return False

# =============================================================================
# 2. CONFIGURAÇÕES GERAIS E ESTADO
# =============================================================================
st.set_page_config(layout="wide", page_title="NeuroStep Analytics")

# --- INICIALIZAÇÃO OBRIGATÓRIA DE ESTADO ---
if 'coletando' not in st.session_state: st.session_state.coletando = False
if 'buffer_dados' not in st.session_state: st.session_state.buffer_dados = np.array([])
if 'start_time' not in st.session_state: st.session_state.start_time = 0.0
if 'tempo_coleta' not in st.session_state: st.session_state.tempo_coleta = 0.0
if 'is_dark_mode' not in st.session_state: st.session_state.is_dark_mode = True 
if 'run_analysis' not in st.session_state: st.session_state.run_analysis = False
if 'paciente_ativo' not in st.session_state: st.session_state.paciente_ativo = None 
if 'mostrar_historico' not in st.session_state: st.session_state.mostrar_historico = False
if 'current_view' not in st.session_state: st.session_state.current_view = 'HOME' 

# --- LÓGICA DE TEMA E CORES ---
eh_dark = st.session_state.is_dark_mode
COR_FUNDO_GRAFICO = '#0E1117' if eh_dark else '#FFF5EE'
COR_TEXTO_GRAFICO = '#E0E0E0' if eh_dark else '#333333'
COR_EIXOS = '#A0A0A0' if eh_dark else '#333333'

# --- CSS DE TEMA (COM AJUSTE DO SELECTBOX LARANJA) ---
if eh_dark:
    CSS_PERSONALIZADO = """
    <style>
        /* --- CONFIGURAÇÕES GERAIS --- */
        .stApp { background-color: #0E1117; color: white; } /* Ajuste para Dark Mode padrão */
        section[data-testid="stSidebar"] { display: none !important; }
        div[data-testid="stButton"] button { border: none !important; }
        
        /* --- ESTILO DOS CARDS DE MÉTRICAS (NOVO) --- */
        div.metric-card {
            background-color: #1E2129; 
            border: 1px solid #333;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
            text-align: center;
            margin-bottom: 10px;
            height: 100%; /* Para alinhar alturas */
        }
        div.metric-card h2 { color: #00FFC8; margin: 0; font-size: 28px; }
        div.metric-card p { color: #A0A0A0; margin: 0; font-size: 14px; }

        /* --- OUTROS ESTILOS JÁ EXISTENTES --- */
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #00FFC8 !important; color: black !important; font-weight: bold;
            box-shadow: 0 0 10px rgba(0, 255, 200, 0.4); transition: 0.3s;
        }
        .main-image-container img { display: block; margin-left: auto; margin-right: auto; }
    </style>
    """
else:
    CSS_PERSONALIZADO = """
    <style>
        /* Fundo Seashell */
        .stApp { background-color: #FFF5EE; color: black; }
        section[data-testid="stSidebar"] { display: none !important; }
        
        h1, h2, h3, h4, h5, h6, p, span, div { color: #333333; }
        div[data-testid="stButton"] button { border: none !important; } 

        /* Métricas em Laranja Vibrante */
        [data-testid="stMetricValue"] { color: #FF751F !important; }
        [data-testid="stMetricLabel"] { color: #666 !important; }
        
        button[key="logo_home_btn"] { font-size: 36px; background: none !important; padding-bottom: 20px; }
        
        /* Botão Primário */
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(90deg, #FF751F 0%, #FE9F5D 100%) !important;
            color: white !important; font-weight: bold;
            box-shadow: 0px 4px 12px rgba(255, 117, 31, 0.3);
        }
        /* Botão Secundário */
        div[data-testid="stButton"] button[kind="secondary"] {
            background-color: #FFBD59 !important; color: #333333 !important; font-weight: 600;
        }
        
        /* 🎯 NOVO: Selectbox (Paciente Ativo) Laranja */
        div[data-testid="stSelectbox"] > div > div {
            background-color: #FFBD59 !important;
            color: #333333 !important;
            border: 1px solid #FF751F !important;
            border-radius: 8px;
        }
        /* Cor do texto dentro do selectbox */
        div[data-testid="stSelectbox"] label {
            color: #FF751F !important; font-weight: bold;
        }
        
        .main-image-container img { display: block; margin-left: auto; margin-right: auto; }
    </style>
    """
st.markdown(CSS_PERSONALIZADO, unsafe_allow_html=True)


# =============================================================================
# 3. CABEÇALHO E BARRA DE CONTROLES (NOVA DISPOSIÇÃO)
# =============================================================================
df_pacientes = carregar_pacientes()
lista_nomes_existentes = df_pacientes['NOME'].tolist()

# Funções de navegação
def navigate_home():
    st.session_state.mostrar_historico = False
    st.session_state.current_view = 'HOME'
    st.rerun()
    
def navigate_history():
    st.session_state.mostrar_historico = True
    st.session_state.current_view = 'HOME'
    st.rerun()

# 1. Definição dos Caminhos das Imagens
LOGO_CLARO = "neurostep_logo_claro.png"
LOGO_ESCURO = "neurostep_logo_escuro.png"

# 2. Lógica de Seleção do Logo baseada no Tema Atual
# Se estiver no modo escuro, usa o logo escuro (geralmente branco/claro para contraste)
# Se estiver no modo claro, usa o logo claro (geralmente escuro para contraste)
if st.session_state.is_dark_mode:
    logo_atual = LOGO_ESCURO
    cor_titulo = "#FFFFFF" # Texto branco no fundo escuro
else:
    logo_atual = LOGO_CLARO
    cor_titulo = "#333333" # Texto preto no fundo claro

# 3. Renderização
col_logo_main, col_spacer_r = st.columns([2, 1])
with col_logo_main:
    # Verifica se o arquivo da imagem existe antes de tentar mostrar
    if os.path.exists(logo_atual):
        st.markdown('<div class="main-image-container">', unsafe_allow_html=True)
        st.image(logo_atual, width=450)
        st.markdown('</div>', unsafe_allow_html=True) 
    else:
        # Fallback caso a imagem não seja encontrada
        st.markdown(f"<h1 style='text-align: center; color: {cor_titulo} !important;'>🧬 NeuroStep</h1>", unsafe_allow_html=True)

# Título Principal (Ajusta a cor conforme o tema também)
st.markdown(
    f"""
    <h1 style='text-align: center; color: {cor_titulo} !important; margin-top: 10px; font-size: 30px;'> 
        ZEBRA 7000 PRO
    </h1>
    """,
    unsafe_allow_html=True)

st.markdown("---")

# --- 2. BARRA DE CONTROLES HORIZONTAL ---
col_nav_home, col_paciente, col_id, col_novo, col_hist, col_tema = st.columns([1, 3, 2, 1.5, 1.5, 1])

# Botão Home
with col_nav_home:
    # Removido o st.markdown("##### Início")
    if st.button("PÁGINA INICIAL", key="nav_home_btn", use_container_width=True): # Adicionei um ícone para contexto
        navigate_home()

# Seleção de Paciente
with col_paciente:
    paciente_selecionado = st.selectbox(
        'Selecione o Paciente', # O texto existe pro sistema, mas escondemos abaixo
        lista_nomes_existentes,
        index=0 if lista_nomes_existentes else None,
        key='seletor_paciente_ativo',
        label_visibility="collapsed" # <--- ISSO ESCONDE O TÍTULO "PACIENTE ATIVO"
    )
    if paciente_selecionado:
        id_ativo = df_pacientes[df_pacientes['NOME'] == paciente_selecionado]['ID_PACIENTE'].iloc[0]
        st.session_state.paciente_ativo = id_ativo
    else:
        st.session_state.paciente_ativo = None

# Display ID
with col_id:
    # Removido o st.markdown("##### ID")
    if st.session_state.paciente_ativo:
        # Mostra apenas o ID direto
        st.success(f"{st.session_state.paciente_ativo}")
    else:
        st.warning("Sem ID")

# Botão Novo Paciente
with col_novo:
    # Removido o st.markdown("##### Cadastro")
    if st.button("CADASTRAR NOVO PACIENTE", key="nav_new_btn", use_container_width=True):
        st.session_state.current_view = 'CADASTRO'
        st.rerun()

# Botão Histórico
with col_hist:
    # Removido o st.markdown("##### Dados")
    if st.button("HISTÓRICO DE EVOLUÇÃO", key="nav_hist_btn", use_container_width=True):
        navigate_history()

# Botão Tema
with col_tema:
    # Removido o st.markdown("##### Tema")
    icon_next = "CLARO" if st.session_state.is_dark_mode else "ESCURO"
    if st.button(f"{icon_next}", key="nav_theme_btn", use_container_width=True):
        st.session_state.is_dark_mode = not st.session_state.is_dark_mode
        st.rerun() 

st.markdown("---")


# =============================================================================
# 4. FUNÇÃO DE RECEIVER (MODO BATCH)
# =============================================================================
def udp_receiver_fixed_time(duration):
    """Recebe pacotes UDP por uma duração fixa, bloqueando a thread."""
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        udp.bind(("0.0.0.0", UDP_PORT))
        udp.settimeout(0.05) 
        data_buffer = []
        start_time = time.time()
        while (time.time() - start_time) < duration:
            try:
                data, addr = udp.recvfrom(1024)
                parts = data.decode('utf-8').strip().split(',')
                if len(parts) == COLUNAS_ESPERADAS:
                    data_buffer.append([float(x) for x in parts])
            except socket.timeout:
                pass
            except Exception as e:
                 st.sidebar.error(f"Erro de formato: {e}")
                 break
        
    except Exception as e:
        st.error(f"Erro de Socket (Porta {UDP_PORT}): {e}. Verifique o Firewall.")
    finally:
        udp.close()
        
    if len(data_buffer) > 0:
        return np.array(data_buffer), time.time() - start_time
    return np.array([]), 0.0

# =============================================================================
# 5. ROTEADOR DE PÁGINAS (CONTEÚDO PRINCIPAL)
# =============================================================================

# --- 5.1 PÁGINA DE CADASTRO ---
if st.session_state.current_view == 'CADASTRO':
    st.header("Cadastro de Novo Paciente")
    
    with st.form(key='novo_paciente_form', clear_on_submit=True):
        novo_nome = st.text_input("Nome Completo", max_chars=100)
        novo_id = st.text_input("ID Único (Prontuário/CPF)", max_chars=12)
        salvar_cadastro = st.form_submit_button("Salvar Paciente e Voltar", type="primary")

        if salvar_cadastro:
            if novo_nome and novo_id and novo_id not in df_pacientes['ID_PACIENTE'].values:
                novo_paciente = pd.DataFrame({'ID_PACIENTE': [novo_id], 'NOME': [novo_nome], 'DATA_CADASTRO': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]})
                salvar_pacientes(pd.concat([df_pacientes, novo_paciente], ignore_index=True))
                st.session_state.message = f"Paciente {novo_nome} cadastrado com sucesso!"
                st.session_state.current_view = 'HOME'
                st.rerun()
            elif not novo_nome or not novo_id:
                st.error("Preencha Nome e ID para salvar.")
            else:
                st.error("Este ID já existe.")
                
    if st.button("⬅ Cancelar", key="cadastro_cancel_btn"):
        st.session_state.current_view = 'HOME'
        st.rerun()


# --- 5.2 PÁGINA PRINCIPAL (HOME/COLETA/HISTÓRICO) ---
elif st.session_state.current_view == 'HOME':
    
    # Exibe mensagens globais
    if 'message' in st.session_state:
        st.markdown(f"<p style='text-align: center; color: green;'>{st.session_state.message}</p>", unsafe_allow_html=True)
        st.session_state.message = None 
        
    # --- MODO HISTÓRICO ---
    if st.session_state.mostrar_historico:
        st.markdown("## 📈 Análise de Evolução do Paciente")
        
        if st.session_state.paciente_ativo and os.path.exists(ARQUIVO_SESSOES):
            df_full = pd.read_csv(ARQUIVO_SESSOES)
            df_hist = df_full[df_full['ID_PACIENTE'].astype(str) == str(st.session_state.paciente_ativo)].copy()
            
            if len(df_hist) > 0:
                st.info(f"Mostrando {len(df_hist)} sessões para o paciente {paciente_selecionado}.")

                # CONFIGURAÇÃO DA TABELA
                df_hist_display = df_hist.copy()
                df_hist_display.insert(0, 'RAW_TIMESTAMP', df_hist_display['DATA_HORA']) 
                df_hist_display['DATA_HORA'] = pd.to_datetime(df_hist_display['DATA_HORA']).dt.strftime('%d/%m/%Y %H:%M:%S')
                
                st.markdown("#### Gerenciamento de Dados:")
                
                df_editado = st.data_editor(
                    df_hist_display[['DATA_HORA', 'CADENCIA', 'PCT_APOIO', 'RMS', 'MDF']],
                    key="tabela_historico_editor",
                    hide_index=True,
                    use_container_width=True,
                    num_rows="dynamic",
                    column_order=('DATA_HORA', 'CADENCIA', 'PCT_APOIO', 'RMS', 'MDF')
                )

                # LÓGICA DE EXCLUSÃO
                if st.button("✅ SALVAR ALTERAÇÕES (Remover Deletados)", key="salvar_deletes_btn", type="primary"):
                    
                    linhas_originais = len(df_hist_display)
                    linhas_editadas = len(df_editado)
                    
                    if linhas_editadas < linhas_originais:
                        timestamps_mantidos_display = df_editado['DATA_HORA'].tolist()
                        df_removidos_display = df_hist_display[~df_hist_display['DATA_HORA'].isin(timestamps_mantidos_display)]
                        timestamps_raw_a_remover = df_removidos_display['RAW_TIMESTAMP'].tolist()

                        if timestamps_raw_a_remover:
                            if excluir_sessoes(timestamps_raw_a_remover):
                                st.session_state.message = f"{len(timestamps_raw_a_remover)} sessões removidas permanentemente."
                                st.rerun() 
                        else:
                            st.info("Nenhuma linha removida foi confirmada.")
                            st.rerun() 
                    else:
                        st.info("Nenhuma alteração de exclusão detectada.")
                        st.rerun() 
                            
                st.markdown("---")
                if len(df_hist) >= 1: 
                    fig_evolucao = gerar_grafico_evolucao(df_hist, COR_FUNDO_GRAFICO, COR_TEXTO_GRAFICO, COR_EIXOS)
                    st.pyplot(fig_evolucao)
            else:
                st.warning("Nenhum histórico encontrado para este paciente.")
        else:
             st.error("Nenhum dado de sessão encontrado no sistema ou nenhum paciente selecionado.")


    # --- MODO COLETA (PADRÃO) ---
    else:
        col_spacer1, col_btn1, col_spacer2 = st.columns([4, 2, 4])
        pode_iniciar = st.session_state.paciente_ativo is not None and not st.session_state.get('run_analysis', False)

        with col_btn1:
            if st.button(f"▶️ INICIAR COLETA ({DURACAO_COLETA}s)", type="primary", use_container_width=True, disabled=not pode_iniciar):
                st.session_state.run_analysis = True
                st.session_state.buffer_dados = np.array([])
                st.session_state.message = f"Coleta iniciada para ID: {st.session_state.paciente_ativo}"
                st.rerun()

        status = st.empty()
        results_container = st.container()

        if st.session_state.get('run_analysis', False):
            status.info(f"Coletando dados... ID: {st.session_state.paciente_ativo}")
            with st.spinner(f"Aguarde a coleta de {DURACAO_COLETA} segundos."):
                dados_coletados, tempo_total = udp_receiver_fixed_time(duration=DURACAO_COLETA)
            st.session_state.run_analysis = False 
            st.session_state.buffer_dados = dados_coletados
            st.session_state.tempo_coleta = tempo_total
            st.rerun()

        if not st.session_state.get('run_analysis', False) and st.session_state.buffer_dados.size > 50:
            status.success("Processando métricas...")
            arr = st.session_state.buffer_dados
            tempo_total = st.session_state.tempo_coleta
            if tempo_total <= 0.0: tempo_total = len(arr) / 100.0 
            fs_real = len(arr) / tempo_total
            
            sinal_semg_bruto = arr[:, 6]
            parametros_semg, sinal_semg_filtrado, sinal_semg_norm = processar_dados_semg(sinal_semg_bruto, fs_real, REGRAS_SEMG) 
            met_marcha = processar_dados_marcha(arr[:, 0:6], fs_real)
            
            patient_id = st.session_state.paciente_ativo 
            if salvar_dados_sessao(patient_id, met_marcha, parametros_semg, fs_real, tempo_total):
                status.success(f"Sessão Salva! ID: {patient_id}")
            
            fig_final = gerar_relatorio_para_streamlit(
                met_marcha, parametros_semg, sinal_semg_filtrado, fs_real, sinal_semg_norm,
                COR_FUNDO_GRAFICO, COR_TEXTO_GRAFICO, COR_EIXOS
            )
            
            with results_container:
                st.subheader(f"Análise da Sessão Atual")
                st.pyplot(fig_final) 

        elif not st.session_state.get('run_analysis', False):
            if st.session_state.paciente_ativo is None:
                status.error("Selecione um Paciente antes de iniciar.")
            else:
                status.info("Aguardando o clique em INICIAR COLETA.")
