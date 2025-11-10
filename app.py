import streamlit as st
import numpy as np
import pandas as pd # Facilita carregar o CSV
import matplotlib.pyplot as plt

# --- Importar as funções de análise do seu outro arquivo ---
# IMPORTANTE: O Streamlit precisa que seu arquivo G7.py esteja no repositório.
# Vamos assumir que o arquivo se chama G7.py
try:
    # Importa as funções e constantes que o app VAI USAR
    from G7 import (
        processar_dados_marcha, 
        processar_dados_semg, 
        gerar_relatorio_combinado,
        REGRAS_MARCHA, 
        REGRAS_SEMG
    )
except ImportError:
    st.error("Erro crítico: O arquivo 'G7.py' não foi encontrado no repositório. "
             "Faça o upload do G7.py junto com este app.py.")
    st.stop()
except Exception as e:
    st.error(f"Erro ao importar 'G7.py': {e}. "
             "Verifique se o G7.py não tem erros de sintaxe.")
    st.stop()


# --- Configuração da Página do Streamlit ---
st.set_page_config(layout="wide", page_title="Análise de Marcha e sEMG")
st.title("🔬 Plataforma de Análise de Marcha e sEMG")
st.write("""
Esta plataforma utiliza os dados brutos coletados do sEMG e IMU 
para gerar relatórios quantitativos da evolução do paciente.
""")

st.divider()

# --- 1. Upload dos Dados ---
st.header("1. Carregar Dados da Coleta")
uploaded_file = st.file_uploader(
    "Selecione o arquivo de dados brutos (.csv ou .txt) gerado pelo coletor:", 
    type=["csv", "txt"]
)

# --- 2. Parâmetros da Análise ---
st.header("2. Configurar Análise")
taxa_amostragem = st.number_input(
    "Taxa de Amostragem Real (Hz)", 
    min_value=1.0, 
    value=100.0, 
    help="Insira a taxa de amostragem (Hz) real da coleta. "
         "Este valor é crucial para a análise."
)

# --- 3. Executar Análise ---
st.header("3. Gerar Relatório")
if st.button("Processar e Gerar Relatório"):
    if uploaded_file is not None:
        try:
            # Carregar os dados usando pandas (mais robusto)
            dados_df = pd.read_csv(uploaded_file, header=None)
            dados_np = dados_df.values
            
            # Verificar colunas
            if dados_np.shape[1] != 7:
                st.warning(f"Aviso: O arquivo tem {dados_np.shape[1]} colunas, "
                           f"mas o esperado eram 7. A análise pode falhar.")

            st.info("Iniciando processamento... Isso pode levar alguns segundos.")

            # --- Análise da MARCHA ---
            with st.spinner('Analisando Marcha (IMU)...'):
                metricas_marcha = processar_dados_marcha(
                    dados_np, 
                    taxa_amostragem,
                    REGRAS_MARCHA
                )

            # --- Análise do sEMG ---
            with st.spinner('Analisando sEMG...'):
                sinal_semg_bruto = dados_np[:, 6] # Pega a 7ª coluna
                parametros_semg, semg_filtrado, _ = processar_dados_semg(
                    sinal_semg_bruto,
                    taxa_amostragem,
                    REGRAS_SEMG
                )

            st.success("Análise de dados concluída! Gerando relatório visual...")

            # --- Geração do Gráfico ---
            # ATENÇÃO: A função 'gerar_relatorio_combinado' do seu G7.py 
            # salva um arquivo (plt.savefig). Precisamos que ela plote no Streamlit.
            
            # Vamos "enganar" ela para não salvar, mas sim nos dar a figura.
            # Iremos criar a figura aqui e passá-la para a função.
            
            # Recriamos a figura aqui
            fig, (ax_report, ax_boxplot, ax_semg) = plt.subplots(
                3, 1, 
                figsize=(12, 18), 
                gridspec_kw={'height_ratios': [2, 2, 3]}
            )
            
            # Chamamos sua função, mas ela NÃO VAI salvar o arquivo
            # (pois não passamos o nome do arquivo)
            # Ela apenas vai "desenhar" na 'fig' que criamos.
            
            # Esta parte é uma RE-IMPLEMENTAÇÃO da sua função de plotagem.
            # É mais seguro refazê-la aqui.
            
            # --- Plot 1: Texto ---
            (m_apoio, s_apoio, m_balanco, s_balanco, cadencia, assimetria, estabilidade) = (
                metricas_marcha['media_apoio'], metricas_marcha['std_apoio'], 
                metricas_marcha['media_balanco'], metricas_marcha['std_balanco'], 
                metricas_marcha['cadencia'], metricas_marcha['assimetria_apoio'], 
                metricas_marcha['estabilidade']
            )
            
            ax_report.set_title("1. Relatório de Análise da Marcha (IMU)", fontsize=18, weight='bold')
            y_pos = 0.85; x_label = 0.05; x_valor = 0.45; x_interp = 0.7
            
            def plot_metrica(ax, label, valor_str, interp_str, cor='black'):
                nonlocal y_pos
                ax.text(x_label, y_pos, label, ha='left', va='center', fontsize=14, weight='bold')
                ax.text(x_valor, y_pos, valor_str, ha='left', va='center', fontsize=14)
                ax.text(x_interp, y_pos, interp_str, ha='left', va='center', fontsize=12, color=cor, style='italic')
                y_pos -= 0.15 

            val_str = f"{cadencia:.1f} p/min" if not np.isnan(cadencia) else "N/A"
            interp, cor = ("Marcha saudável", "green") if 100 <= cadencia <= 120 else (("Marcha lenta", "orange") if 60 <= cadencia < 100 else ("Marcha atípica", "red"))
            plot_metrica(ax_report, "Cadência:", val_str, interp, cor)

            val_str = f"{m_apoio:.2f} s (± {s_apoio:.2f} s)" if not np.isnan(m_apoio) else "N/A"
            plot_metrica(ax_report, "Tempo de Apoio:", val_str, "", "black")

            val_str = f"{m_balanco:.2f} s (± {s_balanco:.2f} s)" if not np.isnan(m_balanco) else "N/A"
            plot_metrica(ax_report, "Tempo de Balanço:", val_str, "", "black")
            
            # ... (adicionar o resto das métricas se quiser) ...
            ax_report.set_xticks([]); ax_report.set_yticks([]); ax_report.axis('off')


            # --- Plot 2: Boxplot ---
            ax_boxplot.set_title('2. Variabilidade dos Tempos de Passo (MARCHA)', fontsize=16)
            # ... (código do boxplot do G7.py) ...
            data_to_plot = [metricas_marcha['tempos_contato'], metricas_marcha['tempos_balanco']]
            labels = ['Apoio', 'Balanço']
            ax_boxplot.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
            ax_boxplot.set_ylabel('Duração (segundos)')


            # --- Plot 3: sEMG ---
            ax_semg.set_title('3. Sinal sEMG (Filtrado)', fontsize=16)
            tempo = np.arange(len(semg_filtrado)) / taxa_amostragem
            ax_semg.plot(tempo, semg_filtrado, color='darkorange', linewidth=0.8)
            ax_semg.set_xlabel('Tempo [s]'); ax_semg.set_ylabel('Amplitude [u.a.]')
            ax_semg.grid(True)

            # --- Mostrar no Streamlit ---
            plt.tight_layout()
            st.pyplot(fig) # Este é o comando mágico!
            
            # --- Mostrar Métricas (Texto) ---
            st.subheader("Métricas Detalhadas")
            st.write("**Métricas da Marcha (IMU):**")
            st.json(metricas_marcha)
            st.write("**Métricas do sEMG (Normalizado):**")
            st.json(parametros_semg)

        except Exception as e:
            st.error(f"Ocorreu um erro durante a análise: {e}")
            st.exception(e) # Mostra mais detalhes do erro
            
    else:
        st.error("Por favor, faça o upload de um arquivo de dados primeiro.")