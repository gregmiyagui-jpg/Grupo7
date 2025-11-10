import socket
import time
import numpy as np
from scipy.signal import find_peaks, iirnotch, butter, filtfilt
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os

# --- Configurações de Rede ---
HOST_IP = '0.0.0.0' 
HOST_PORT = 4210    

# --- Configurações da Coleta ---
TEMPO_COLETA_SEGUNDOS = 15
COLUNAS_ESPERADAS = 7 

# --- Configurações da Análise de MARCHA (IMU) ---
REGRAS_MARCHA = {
    'prominencia_az_ms2': 5.0,  
    'prominencia_g_rads': 1.0,  
    'periodo_refratario_ms': 200,
    'limite_apoio_s': (0.3, 1.5),
    'limite_balanco_s': (0.2, 1.2)
}

# --- Configurações da Análise de sEMG ---
REGRAS_SEMG = {
    'aplicar_filtro_notch': True,
    'notch_freq': 60.0,
    'quality_factor': 30.0,
    'low_cut': 30.0,
    'high_cut': 450.0,
    'order': 4
}


# =============================================================================
# --- FUNÇÕES DE ANÁLISE DA MARCHA (IMU) ---
# (Esta função não mudou)
# =============================================================================

def processar_dados_marcha(dados_coletados_imu, taxa_amostragem, regras):
    
    print("\nIniciando processamento dos dados da MARCHA...")
    
    dados = np.array(dados_coletados_imu)
    ax_signal = dados[:, 0]
    ay_signal = dados[:, 1]
    az_signal = dados[:, 2] 
    g_sinal_sagital = dados[:, 3] 
    
    dt = 1.0 / taxa_amostragem
    
    thr_az = regras['prominencia_az_ms2']
    thr_g_sagital = regras['prominencia_g_rads']
    refractory_samples = int((regras['periodo_refratario_ms'] / 1000.0) / dt)
    
    print(f"Taxa de amostragem REAL (Marcha): {taxa_amostragem:.2f} Hz (dt={dt*1000:.1f} ms)")
    print(f"Limiar Fixo 'az' (Prominência): {thr_az:.2f} m/s^2")
    print(f"Limiar Fixo 'g_sagital' (Prominência): {thr_g_sagital:.2f} rad/s")
    print(f"Período Refratário: {regras['periodo_refratario_ms']} ms ({refractory_samples} amostras)")

    tc_indices, _ = find_peaks(az_signal, prominence=thr_az)
    ti_indices, _ = find_peaks(g_sinal_sagital, prominence=thr_g_sagital)
    
    eventos_tc_validos = []
    janela_busca_amostras = 5 
    for idx in tc_indices:
        inicio_busca = max(1, idx - janela_busca_amostras) 
        fim_busca = idx + 1 
        encontrou_zero_cross = False
        for i in range(inicio_busca, fim_busca):
            if g_sinal_sagital[i-1] < 0 and g_sinal_sagital[i] >= 0:
                encontrou_zero_cross = True
                break 
        if encontrou_zero_cross:
            eventos_tc_validos.append(idx)
            
    print(f"Candidatos 'tc' (picos em Az): {len(tc_indices)}")
    print(f"Candidatos 'ti' (picos em Gx): {len(ti_indices)}")
    print(f"Eventos 'tc' válidos (com zero-cross de Gx): {len(eventos_tc_validos)}")
    
    eventos = []
    for idx in eventos_tc_validos: eventos.append((idx, 'tc'))
    for idx in ti_indices: eventos.append((idx, 'ti'))
    eventos.sort(key=lambda x: x[0]) 

    eventos_finais = []
    ultimo_idx = -np.inf
    tipo_esperado = 'tc'
    for idx, tipo in eventos:
        if (idx - ultimo_idx) < refractory_samples: continue
        if tipo == tipo_esperado:
            eventos_finais.append((idx, tipo))
            ultimo_idx = idx
            tipo_esperado = 'ti' if tipo == 'tc' else 'tc'

    tempos_de_contato = []
    tempos_de_balanco = []
    lim_apoio_min, lim_apoio_max = regras['limite_apoio_s']
    lim_bal_min, lim_bal_max = regras['limite_balanco_s']
    for i in range(len(eventos_finais) - 1):
        idx_1, tipo_1 = eventos_finais[i]
        idx_2, tipo_2 = eventos_finais[i+1]
        duracao_s = (idx_2 - idx_1) * dt
        if tipo_1 == 'tc' and tipo_2 == 'ti':
            if lim_apoio_min <= duracao_s <= lim_apoio_max:
                tempos_de_contato.append(duracao_s)
        elif tipo_1 == 'ti' and tipo_2 == 'tc':
            if lim_bal_min <= duracao_s <= lim_bal_max:
                tempos_de_balanco.append(duracao_s)
    
    cadencia = np.nan
    indices_tc_finais = [idx for idx, tipo in eventos_finais if tipo == 'tc']
    if len(indices_tc_finais) > 1:
        intervalos_tc_s = np.diff(indices_tc_finais) * dt
        intervalo_medio_tc_s = np.mean(intervalos_tc_s)
        if intervalo_medio_tc_s > 0:
            cadencia = 60.0 / intervalo_medio_tc_s

    assimetria_apoio = np.nan
    media_apoio = np.nan
    std_apoio = np.nan
    if len(tempos_de_contato) > 1:
        media_apoio = np.mean(tempos_de_contato)
        std_apoio = np.std(tempos_de_contato)
        if media_apoio > 0:
            assimetria_apoio = (std_apoio / media_apoio) * 100

    media_balanco = np.nan
    std_balanco = np.nan
    if len(tempos_de_balanco) > 1:
        media_balanco = np.mean(tempos_de_balanco)
        std_balanco = np.std(tempos_de_balanco)

    estabilidade = np.nan
    accel_mag = np.sqrt(ax_signal*2 + ay_signal2 + az_signal*2)
    media_mag = np.mean(accel_mag)
    std_mag = np.std(accel_mag)
    if media_mag > 0:
        estabilidade = std_mag / media_mag
        
    metricas = {
        'tempos_contato': tempos_de_contato,
        'tempos_balanco': tempos_de_balanco,
        'media_apoio': media_apoio,
        'std_apoio': std_apoio,
        'media_balanco': media_balanco,
        'std_balanco': std_balanco,
        'cadencia': cadencia,
        'assimetria_apoio': assimetria_apoio,
        'estabilidade': estabilidade
    }
    return metricas

# =============================================================================
# --- FUNÇÕES DE ANÁLISE DE sEMG ---
# (calculate_emg_parameters, notch_filter, butter_bandpass_filter não mudaram)
# =============================================================================

def notch_filter(data, freq, Q, fs):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if high >= 1.0:
        print(f"Aviso: 'high_cut' ({highcut} Hz) está acima ou muito perto de Nyquist ({nyq} Hz).")
        print(f"Ajustando 'high_cut' para {nyq * 0.99:.2f} Hz.")
        high = 0.99 
        
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_emg_parameters(data, fs):
    params = {}
    params['RMS'] = np.sqrt(np.mean(data**2))
    params['MAV'] = np.mean(np.abs(data))
    params['WL'] = np.sum(np.abs(np.diff(data)))
    params['LOG'] = np.exp(np.mean(np.log(np.abs(data) + 1e-10)))
    N = len(data)
    yf = fft(data)
    psd = (1/(N*fs)) * np.abs(yf[0:N//2])**2
    xf = fftfreq(N, 1 / fs)[:N//2]
    
    sum_psd = np.sum(psd)
    if sum_psd == 0:
        params['MNF'] = 0.0
        params['MDF'] = 0.0
        return params
    params['MNF'] = np.sum(xf * psd) / sum_psd
    cumulative_power = np.cumsum(psd)
    total_power = sum_psd
    median_freq_index = np.where(cumulative_power >= total_power / 2)[0]
    
    if len(median_freq_index) > 0:
        params['MDF'] = xf[median_freq_index[0]]
    else:
        params['MDF'] = 0.0
    return params

# --- [MUDANÇA AQUI] ---
def processar_dados_semg(sinal_bruto, fs, configs):
    """
    Encapsula o pipeline de processamento do sEMG.
    [V4] Usa Normalização por Pico (divisão pelo máx. absoluto).
    """
    print(f"\nIniciando processamento dos dados de sEMG...")
    print(f"Taxa de amostragem REAL (sEMG): {fs:.2f} Hz")
    
    # 1. Filtragem
    dados_intermediarios = sinal_bruto
    if configs['aplicar_filtro_notch']:
        print(f"Aplicando Notch Filter @ {configs['notch_freq']} Hz")
        dados_intermediarios = notch_filter(
            dados_intermediarios, configs['notch_freq'], configs['quality_factor'], fs
        )
    print(f"Aplicando Bandpass Filter @ {configs['low_cut']}-{configs['high_cut']} Hz")
    dados_filtrados = butter_bandpass_filter(
        dados_intermediarios, configs['low_cut'], configs['high_cut'], fs, configs['order']
    )

    # 2. Normalização (MÉTODO ALTERADO: Divisão pelo Máximo Absoluto)
    print("Aplicando Normalização por Pico (divisão pelo valor máximo absoluto)...")
    
    # Encontra o valor de pico absoluto no sinal filtrado
    max_abs_val = np.max(np.abs(dados_filtrados))
    
    if max_abs_val == 0:
        # Evita divisão por zero se o sinal for completamente plano
        dados_normalizados = dados_filtrados 
        print("Aviso: O valor máximo do sinal é zero. Normalização não aplicada.")
    else:
        # Divide todo o sinal (positivo e negativo) por esse pico
        dados_normalizados = dados_filtrados / max_abs_val
        print("Normalização por Pico aplicada.")
    
    # 3. Extração de Parâmetros
    print("Extraindo parâmetros do sinal normalizado...")
    parametros_calculados = calculate_emg_parameters(dados_normalizados, fs)
    
    # Retorna os dados para plotagem e parâmetros
    # (Note que retornamos 'dados_filtrados' para o gráfico, 
    # e 'parametros_calculados' da versão normalizada)
    return parametros_calculados, dados_filtrados, dados_normalizados

# =============================================================================
# --- FUNÇÃO DE RELATÓRIO COMBINADO ---
# (Esta função não mudou)
# =============================================================================

def gerar_relatorio_combinado(metricas_marcha, sinal_semg_filtrado, fs, nome_arquivo):
    """
    Cria um único arquivo PNG com 3 subplots:
    1. Relatório de texto da Marcha (IMU)
    2. Boxplot de variabilidade da Marcha (IMU)
    3. Gráfico do sinal de sEMG filtrado
    """
    print(f"\nGerando relatório combinado: {nome_arquivo}")

    fig, (ax_report, ax_boxplot, ax_semg) = plt.subplots(
        3, 1, 
        figsize=(12, 18), 
        gridspec_kw={'height_ratios': [2, 2, 3]}
    )
    
    fig.patch.set_facecolor('#f4f4f4')
    fig.suptitle("Relatório Combinado de Análise da Marcha e sEMG", fontsize=24, weight='bold')

    # --- 1. Plot 1: Relatório de Texto do IMU ---
    m_apoio = metricas_marcha['media_apoio']
    s_apoio = metricas_marcha['std_apoio']
    m_balanco = metricas_marcha['media_balanco']
    s_balanco = metricas_marcha['std_balanco']
    cadencia = metricas_marcha['cadencia']
    assimetria = metricas_marcha['assimetria_apoio']
    estabilidade = metricas_marcha['estabilidade']
    
    ax_report.set_title("1. Relatório de Análise da Marcha (IMU)", fontsize=18, weight='bold')
    
    y_pos = 0.85 
    x_label = 0.05
    x_valor = 0.45
    x_interp = 0.7
    
    def plot_metrica(ax, label, valor_str, interp_str, cor='black'):
        nonlocal y_pos
        ax.text(x_label, y_pos, label, ha='left', va='center', fontsize=14, weight='bold')
        ax.text(x_valor, y_pos, valor_str, ha='left', va='center', fontsize=14)
        ax.text(x_interp, y_pos, interp_str, ha='left', va='center', fontsize=12, color=cor, style='italic')
        y_pos -= 0.15 

    val_str = f"{cadencia:.1f} passos/min" if not np.isnan(cadencia) else "N/A"
    interp, cor = "", "#333333"
    if not np.isnan(cadencia):
        if 100 <= cadencia <= 120: interp, cor = "Marcha saudável", "green"
        elif 60 <= cadencia < 100: interp, cor = "Marcha lenta/reduzida", "orange"
        else: interp, cor = "Marcha atípica", "red"
    plot_metrica(ax_report, "Cadência:", val_str, interp, cor)

    val_str = f"{m_apoio:.2f} s  (± {s_apoio:.2f} s)" if not np.isnan(m_apoio) else "N/A"
    plot_metrica(ax_report, "Tempo de Apoio (Média ± DP):", val_str, "", "black")

    val_str = f"{m_balanco:.2f} s  (± {s_balanco:.2f} s)" if not np.isnan(m_balanco) else "N/A"
    plot_metrica(ax_report, "Tempo de Balanço (Média ± DP):", val_str, "", "black")

    val_str = f"{assimetria:.1f} % (CV)" if not np.isnan(assimetria) else "N/A"
    interp, cor = "", "#333333"
    if not np.isnan(assimetria):
        if assimetria <= 5: interp, cor = "Marcha simétrica", "green"
        elif 5 < assimetria <= 15: interp, cor = "Leve assimetria", "orange"
        else: interp, cor = "Assimetria significativa", "red"
    plot_metrica(ax_report, "Variabilidade do Apoio:", val_str, interp, cor)
    
    val_str = f"{estabilidade:.2f} (CV)" if not np.isnan(estabilidade) else "N/A"
    interp, cor = "", "#333333"
    if not np.isnan(estabilidade):
        if estabilidade < 0.20: interp, cor = "Marcha estável", "green"
        elif 0.20 <= estabilidade <= 0.40: interp, cor = "Leve instabilidade", "orange"
        else: interp, cor = "Instabilidade alta", "red"
    plot_metrica(ax_report, "Estabilidade (Geral):", val_str, interp, cor)

    ax_report.set_xticks([])
    ax_report.set_yticks([])
    ax_report.spines['top'].set_visible(False)
    ax_report.spines['right'].set_visible(False)
    ax_report.spines['bottom'].set_visible(False)
    ax_report.spines['left'].set_visible(False)
    ax_report.set_facecolor('#f4f4f4')

    # --- 2. Plot 2: Boxplot da Marcha ---
    ax_boxplot.set_title('2. Variabilidade dos Tempos de Passo (MARCHA)', fontsize=16, weight='bold')
    t_apoio = metricas_marcha['tempos_contato']
    t_balanco = metricas_marcha['tempos_balanco']
    data_to_plot = []
    labels = []
    
    if len(t_apoio) > 1:
        data_to_plot.append(t_apoio)
        labels.append(f'Apoio (Contato)\n(n={len(t_apoio)})') 
    if len(t_balanco) > 1:
        data_to_plot.append(t_balanco)
        labels.append(f'Balanço (Swing)\n(n={len(t_balanco)})')
        
    if data_to_plot:
        box_parts = ax_boxplot.boxplot(data_to_plot, 
                               labels=labels, 
                               patch_artist=True, 
                               showmeans=True,  
                               medianprops={'color': 'black', 'linewidth': 2}, 
                               meanprops={'marker':'o', 'markeredgecolor':'black', 
                                          'markerfacecolor':'white'} 
                               ) 
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(box_parts['boxes'], colors):
            patch.set_facecolor(color)
        ax_boxplot.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax_boxplot.set_ylabel('Duração (segundos)', fontsize=12)
        ax_boxplot.set_xticklabels(labels, fontsize=12)
    else:
        ax_boxplot.text(0.5, 0.5, "N/A (Passos insuficientes para o boxplot)", 
                        ha='center', va='center', fontsize=14, color='gray')
        ax_boxplot.set_xticks([])
        ax_boxplot.set_yticks([])

    # --- 3. Plot 3: Gráfico do sEMG Filtrado ---
    ax_semg.set_title('3. Sinal sEMG (Após Filtragem Completa)', fontsize=16, weight='bold')
    tempo = np.arange(len(sinal_semg_filtrado)) / fs
    
    ax_semg.plot(tempo, sinal_semg_filtrado, color='darkorange', linewidth=0.8)
    ax_semg.set_xlabel('Tempo [s]', fontsize=12)
    ax_semg.set_ylabel('Amplitude [u.a.]')
    ax_semg.grid(True)
    ax_semg.set_xlim(left=0, right=max(tempo))

    # --- Salvar o arquivo final ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(nome_arquivo, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# =================================================================
# SCRIPT PRINCIPAL (Coleta e Chamada das Análises)
# (Esta seção não mudou)
# =================================================================

dados_coletados = []
print(f"[REDE] Iniciando servidor UDP em {HOST_IP}:{HOST_PORT}")
print(f"Aguardando dados do ESP32... (você tem {TEMPO_COLETA_SEGUNDOS}s após o primeiro pacote)")
tempo_total_coleta = 0.0
coleta_iniciada = False
tempo_inicio = 0

try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((HOST_IP, HOST_PORT))
        sock.settimeout(3.0) 

        while True:
            try:
                linha_bytes, addr = sock.recvfrom(1024)
                
                if not coleta_iniciada:
                    print(f"Primeiro pacote recebido de {addr}! Iniciando coleta de {TEMPO_COLETA_SEGUNDOS}s...")
                    tempo_inicio = time.time()
                    coleta_iniciada = True

                if coleta_iniciada and (time.time() - tempo_inicio) > TEMPO_COLETA_SEGUNDOS:
                    print("Tempo de coleta esgotado.")
                    break 

                linha_str = ""
                try:
                    linha_str = linha_bytes.decode('utf-8').strip()
                    if linha_str:
                        valores_str = linha_str.split(',')
                        
                        if len(valores_str) == COLUNAS_ESPERADAS: 
                            valores_float = [float(v) for v in valores_str]
                            dados_coletados.append(valores_float)
                        else:
                            print(f"Pacote ignorado (formato inválido, esperava {COLUNAS_ESPERADAS} colunas): '{linha_str}'")
                            
                except Exception as e:
                    if linha_str: 
                        print(f"Pacote ignorado (erro ao decodificar): '{linha_str}'")
                    pass 
            
            except socket.timeout:
                if coleta_iniciada:
                    print("Timeout: Dados pararam de chegar. Encerrando coleta.")
                    break 
                else:
                    print("Aguardando o primeiro pacote...")
                    pass 

        if coleta_iniciada:
            tempo_fim = time.time()
            tempo_total_coleta = tempo_fim - tempo_inicio
        else:
            tempo_total_coleta = 0

    # --- Fim da Coleta ---
    print("\n-------------------------------------------")
    print(f"Coleta (via Wi-Fi) concluída.")
    if tempo_total_coleta:
         print(f"Tempo total real de coleta: {tempo_total_coleta:.2f} segundos.")
    total_amostras = len(dados_coletados)
    print(f"Total de amostras válidas coletadas: {total_amostras}")
    print("-------------------------------------------\n")

    if total_amostras > 0:
        # Salva os dados brutos em um arquivo de texto
        i = 1
        while os.path.exists(f"coleta_bruta_{i}.csv"):
            i += 1
        nome_arquivo_bruto = f"coleta_bruta_{i}.csv"
        
        try:
            np.savetxt(nome_arquivo_bruto, dados_coletados, delimiter=",")
            print(f"DADOS BRUTOS SALVOS EM: {nome_arquivo_bruto}")
        except Exception as e:
            print(f"Erro ao salvar dados brutos: {e}")
            
    # #################################################
    # <<< CHAMADA PARA AMBAS AS ANÁLISES >>>
    # #################################################
    
    if total_amostras > 50:
        
        if tempo_total_coleta == 0:
            print("Erro: Tempo de coleta foi zero. Impossível calcular taxa de amostragem.")
            raise Exception("Tempo de coleta nulo.")
            
        taxa_amostragem_real = total_amostras / tempo_total_coleta
        
        dados_np = np.array(dados_coletados)

        # --- 1. Processamento da MARCHA (IMU) ---
        metricas_marcha = processar_dados_marcha(
            dados_np, 
            taxa_amostragem_real,
            REGRAS_MARCHA
        )
        
        # --- 2. Processamento do sEMG ---
        sinal_semg_bruto = dados_np[:, 6] 
        # A função agora usa a nova normalização internamente
        parametros_semg, semg_filtrado, semg_normalizado = processar_dados_semg(
            sinal_semg_bruto,
            taxa_amostragem_real,
            REGRAS_SEMG
        )
        
        # --- 3. Relatórios no Console ---
        print("\n--- RELATÓRIO DE TEMPOS DA MARCHA (Console) ---")
        if metricas_marcha['tempos_contato']:
            print("Tempos de Contato (Apoio) [s]:", [round(t, 2) for t in metricas_marcha['tempos_contato']])
        else:
            print("Tempos de Contato (Apoio) [s]: Nenhum passo detectado.")
        if metricas_marcha['tempos_balanco']:
            print("Tempos de Balanço (Swing) [s]:", [round(t, 2) for t in metricas_marcha['tempos_balanco']])
        else:
            print("Tempos de Balanço (Swing) [s]: Nenhum passo detectado.")
        
        print("\n--- RELATÓRIO DE MÉTRICAS AVANÇADAS (MARCHA - Console) ---")
        print(f"Cadência: {metricas_marcha['cadencia']:.1f} passos/min" if not np.isnan(metricas_marcha['cadencia']) else "Cadência: N/A (Passos insuficientes)")
        print(f"Variabilidade (Apoio): {metricas_marcha['assimetria_apoio']:.1f} %" if not np.isnan(metricas_marcha['assimetria_apoio']) else "Variabilidade (Apoio): N/A (Passos insuficientes)")
        print(f"Estabilidade (Geral): {metricas_marcha['estabilidade']:.2f}" if not np.isnan(metricas_marcha['estabilidade']) else "Estabilidade (Geral): N/A")
        
        print("\n--- RELATÓRIO DE PARÂMETROS sEMG (Console) ---")
        print("(Calculados sobre o sinal normalizado por pico)")
        for nome, valor in parametros_semg.items():
            print(f"{nome}: {valor:.4f}")
        print("-------------------------------------------------")
            
        
        # --- 4. Geração de Gráfico COMBINADO ---
        i = 1
        while os.path.exists(f"relatorio_combinado_{i}.png"):
            i += 1
        nome_relatorio_final = f"relatorio_combinado_{i}.png"

        try:
            gerar_relatorio_combinado(
                metricas_marcha,
                semg_filtrado,          
                taxa_amostragem_real,   
                nome_relatorio_final
            )
            print(f"\nProcesso de análise concluído. Verifique o arquivo: {nome_relatorio_final}")
            
        except Exception as e:
            print(f"\nErro ao gerar o relatório combinado: {e}")
            print("Verifique se a biblioteca 'matplotlib' está instalada.")

    else:
        print("Não foram coletados dados suficientes para a análise (menos de 50 amostras).")

except socket.error as e:
    print(f"Erro de Socket: {e}")
    print(f"Verifique se o IP/Porta estão corretos e se o Firewall não está bloqueando a porta {HOST_PORT}.")
except Exception as e:

    print(f"Ocorreu um erro inesperado: {e}")
