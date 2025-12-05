# -- coding: utf-8 --
"""
SCRIPT FINAL DE ANÁLISE: MARCHA INTELIGENTE + sEMG COMPLETO (Biblioteca)
Versão com Envelopamento Linear (Sugestão da Professora) mantendo dados originais.
"""

import socket
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, iirnotch, butter, filtfilt
from scipy.fft import fft, fftfreq
import os

# --- CONFIGURAÇÕES ---
REGRAS_SEMG = {
    'aplicar_filtro_notch': True, 
    'notch_freq': 60.0, 
    'quality_factor': 30.0,
    'low_cut': 30.0, 
    'high_cut': 450.0, 
    'order': 4
}
NOMES_PARAMETROS_SEMG = {
    'RMS': 'Constância da força de contração (RMS)', 
    'MAV': 'Média da amplitude absoluta (MAV)', 
    'LOG': 'Intensidade da força de contração (LOG)',
    'WL':  'Complexidade do sinal (WL)', 
    'MNF': 'Freq. Média (MNF)', 
    'MDF': 'Freq. Mediana (MDF)'
}

# =============================================================================
# --- FUNÇÕES AUXILIARES ---
# =============================================================================

def butter_lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    if nyq <= cutoff: return data 
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0: normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low >= 1.0 or low >= high: return data 
    if high >= 1.0: high = 0.99 
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, freq, Q, fs):
    nyq = 0.5 * fs
    w0 = freq / nyq
    if w0 >= 1.0: return data 
    b, a = iirnotch(w0, Q) 
    return filtfilt(b, a, data)

def processar_dados_marcha(dados_brutos, fs):
    
    # (LÓGICA MANTIDA IDÊNTICA AO ORIGINAL)
    df = pd.DataFrame(dados_brutos[:, :6], columns=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])
    tempo = np.arange(len(df)) / fs
    std_g = df[['Gx', 'Gy', 'Gz']].std()
    eixo_dominante = std_g.idxmax()
    sinal_raw = df[eixo_dominante].values
    sinal_filt = butter_lowpass(sinal_raw, 3, fs)
    
    if np.abs(np.min(sinal_filt)) > np.max(sinal_filt): 
        sinal_processamento = -sinal_filt
    else: 
        sinal_processamento = sinal_filt
        
    altura_min = np.max(sinal_processamento) * 0.25 
    picos, _ = find_peaks(sinal_processamento, height=altura_min, distance=fs*0.5)
    
    hs_times = []; 
    to_times = []; 
    hs_ind = []; 
    to_ind = []; 
    janela = int(0.4 * fs)
    
    for p in picos:
        inicio = max(0, p - janela); 
        segmento_pre = sinal_processamento[inicio:p]
        if len(segmento_pre) > 0:
            idx_real = inicio + np.argmin(segmento_pre); 
            to_times.append(tempo[idx_real]); 
            to_ind.append(idx_real)
            
        fim = min(len(sinal_processamento), p + janela); 
        segmento_pos = sinal_processamento[p:fim]
        if len(segmento_pos) > 0:
            idx_real = p + np.argmin(segmento_pos); 
            hs_times.append(tempo[idx_real]); 
            hs_ind.append(idx_real)
            
    media_apoio = np.nan; 
    media_balanco = np.nan; 
    cadencia = np.nan; 
    pct_apoio = np.nan; 
    estabilidade = np.nan
    
    if len(hs_times) > 1:
        stride_durations = np.diff(hs_times);
        media_stride = np.mean(stride_durations)
        
        min_len = min(len(hs_times), len(to_times))
        swing_durations = np.array(hs_times[:min_len]) - np.array(to_times[:min_len])
        swing_durations = swing_durations[swing_durations > 0]
        
        if len(swing_durations) > 0 and media_stride > 0:
            media_balanco = np.mean(swing_durations); 
            media_apoio = media_stride - media_balanco
            cadencia = 120.0 / media_stride; 
            pct_apoio = (media_apoio / media_stride) * 100.0
            estabilidade = (np.std(stride_durations) / media_stride) * 100
            
    metricas = {
        'media_apoio': media_apoio, 
        'media_balanco': media_balanco, 
        'cadencia': cadencia,
        'pct_apoio': pct_apoio, 
        'estabilidade': estabilidade, 
        'tempo': tempo,
        'sinal_plot': sinal_filt, 
        'nome_eixo': eixo_dominante, 
        'hs_ind': np.array(hs_ind),
        'to_ind': np.array(to_ind), 
        'picos_ind': picos
    }
    return metricas

def calculate_emg_parameters(data, fs):
    # (LÓGICA MANTIDA IDÊNTICA AO ORIGINAL)
    params = {}
    data_safe = np.abs(data) + 1e-10
    
    params['RMS'] = np.sqrt(np.mean(data**2)); 
    params['MAV'] = np.mean(np.abs(data));
    params['LOG'] = np.exp(np.mean(np.log(data_safe))); 
    params['WL']  = np.sum(np.abs(np.diff(data)));
    
    N = len(data); 
    yf = fft(data); 
    psd = (1/(N*fs)) * np.abs(yf[0:N//2])**2; 
    xf = fftfreq(N, 1 / fs)[:N//2]
    
    sum_psd = np.sum(psd)
    if sum_psd > 0:
        params['MNF'] = np.sum(xf * psd) / sum_psd
        cumulative_power = np.cumsum(psd)
        median_freq_index = np.where(cumulative_power >= sum_psd / 2)[0]
        params['MDF'] = xf[median_freq_index[0]] if len(median_freq_index) > 0 else 0
    else: 
        params['MNF'] = 0; 
        params['MDF'] = 0
    return params

def processar_dados_semg(sinal_bruto, fs, configs):
    """
    Processa o sEMG e gera o Envelope, mas calcula métricas no sinal filtrado padrão.
    """
    # 1. Filtros Padrão (Notch + Bandpass)
    dados = sinal_bruto
    if configs['aplicar_filtro_notch']: 
        dados = notch_filter(dados, configs['notch_freq'], configs['quality_factor'], fs)
    dados_filtrados = butter_bandpass_filter(dados, configs['low_cut'], configs['high_cut'], fs, configs['order'])
    
    # 2. Normalização
    mx = np.max(np.abs(dados_filtrados))
    dados_norm = dados_filtrados / mx if mx > 0 else dados_filtrados
    
    # 3. Cálculo dos Parâmetros (NO DADO NORMALIZADO ORIGINAL - PRESERVA A MATEMÁTICA DO GRUPO)
    params = calculate_emg_parameters(dados_norm, fs)
    
    # 4. Geração do Envelope (Apenas para Visualização)
    # Retifica e aplica LowPass de 5Hz
    envelope = butter_lowpass(np.abs(dados_norm), 5.0, fs, order=4)
    
    # Retorna: params, sinal_filtrado (para debug), envelope (para plot bonito)
    return params, dados_filtrados, envelope

# =============================================================================
# --- 3. FUNÇÃO DE PLOTAGEM PARA STREAMLIT (COM ENVELOPE) ---
# =============================================================================

def gerar_relatorio_para_streamlit(metricas, params_semg, sinal_semg_filtrado, fs, sinal_envelope, cor_fundo, cor_texto, cor_eixos):
    fig, axs = plt.subplots(2, 2, figsize=(18, 14), facecolor=cor_fundo)
    fig.patch.set_facecolor(cor_fundo)
    fig.suptitle("Análise Biomecânica: Marcha & Eletromiografia", fontsize=26, weight='bold', color=cor_texto)

    # 1. BLOCO TEXTO IMU
    ax_txt_imu = axs[0, 0]; 
    ax_txt_imu.axis('off')
    ax_txt_imu.set_title("1. Relatório da Marcha", fontsize=18, weight='bold', color=cor_texto)
    
    cad = metricas['cadencia']; 
    apoio_pct = metricas['pct_apoio']
    diagnostico = "Indefinido"; 
    cor_diag = "gray"
    
    if not np.isnan(cad):
        if cad > 85 and apoio_pct < 66: 
            diagnostico, cor_diag = "SAUDÁVEL/NORMAL", "green"
        elif cad < 60: 
            diagnostico, cor_diag = "ATÍPICA (Lenta)", "red"
        elif apoio_pct > 70: 
            diagnostico, cor_diag = "ATÍPICA (Instável)", "red"
        else: 
            diagnostico, cor_diag = "LEVEMENTE ANORMAL", "orange"

    y = 0.85
    def pt(ax, l, v, c=cor_texto): 
        nonlocal y
        ax.text(0.05, y, l, fontsize=14, weight='bold', color=cor_texto); 
        ax.text(0.6, y, v, fontsize=14, color=c)
        y -= 0.12

    pt(ax_txt_imu, "Cadência:", f"{cad:.1f} ppm" if not np.isnan(cad) else "N/A")
    pt(ax_txt_imu, "Tempo Apoio:", f"{metricas['media_apoio']:.3f} s" if not np.isnan(metricas['media_apoio']) else "N/A")
    pt(ax_txt_imu, "Tempo Balanço:", f"{metricas['media_balanco']:.3f} s" if not np.isnan(metricas['media_balanco']) else "N/A")
    pt(ax_txt_imu, "% do Ciclo em Apoio:", f"{apoio_pct:.1f} %" if not np.isnan(apoio_pct) else "N/A")
    
    y -= 0.05
    ax_txt_imu.text(0.05, y, "DIAGNÓSTICO:", fontsize=14, weight='bold')
    ax_txt_imu.text(0.05, y-0.1, diagnostico, fontsize=20, weight='bold', color=cor_diag, bbox=dict(facecolor='white', alpha=0.9, edgecolor=cor_diag))

    # 2. BLOCO TEXTO sEMG
    ax_txt_semg = axs[0, 1]; 
    ax_txt_semg.axis('off')
    ax_txt_semg.set_title("2. Métricas Musculares", fontsize=18, weight='bold', color=cor_texto)
    y_semg = 0.90
    
    for k, v_nome in NOMES_PARAMETROS_SEMG.items():
        ax_txt_semg.text(0.05, y_semg, f"{v_nome}:", fontsize=13, weight='bold', color=cor_texto)
        ax_txt_semg.text(0.6, y_semg, f"{params_semg.get(k,0):.4f}", fontsize=13, color=cor_texto)
        y_semg -= 0.10

    # 3. PLOT IMU
    ax_im = axs[1, 0]; 
    ax_im.set_facecolor(cor_fundo)
    t = metricas['tempo']; 
    s = metricas['sinal_plot']
    ax_im.plot(t, s, color=cor_texto, alpha=0.6, label='Giroscópio')
    
    pk = metricas['picos_ind']
    hs = metricas['hs_ind']; 
    to = metricas['to_ind']
    
    if len(pk)>0: ax_im.plot(t[pk], s[pk], 'rx', label='Mid-Swing')
    if len(hs)>0: ax_im.plot(t[hs], s[hs], 'bo', label='Heel Strike')
    if len(to)>0: ax_im.plot(t[to], s[to], 'go', label='Toe Off')
    
    ax_im.set_title(f"Cinemática ({metricas['nome_eixo']})", color=cor_texto)
    ax_im.legend(facecolor=cor_fundo, labelcolor=cor_texto)
    ax_im.grid(True, linestyle='--', alpha=0.3)
    ax_im.tick_params(colors=cor_texto); ax_im.xaxis.label.set_color(cor_texto); ax_im.yaxis.label.set_color(cor_texto)

    # 4. PLOT sEMG (COM ENVELOPE E RAW NO FUNDO)
    ax_emg = axs[1, 1]; ax_emg.set_facecolor(cor_fundo)
    ts = np.arange(len(sinal_envelope))/fs
    
    # Fundo: Sinal Filtrado (Original) - Leve e transparente
    # Precisamos recalcular o sinal norm original rapidinho para plotar no fundo
    mx = np.max(np.abs(sinal_semg_filtrado)); 
    s_orig_norm = sinal_semg_filtrado/mx if mx>0 else sinal_semg_filtrado
    ax_emg.plot(ts, s_orig_norm, color='white', alpha=0.3, lw=0.5, label='Sinal Bruto (Filtrado)')
    
    # Frente: Envelope (Sugestão da Professora) - Grosso e colorido
    ax_emg.plot(ts, sinal_envelope, color='#ff7f0e', lw=2.0, label='Envelope Linear (5Hz)')
    
    ax_emg.set_title("Ativação Muscular (Envelope)", color=cor_texto)
    ax_emg.set_xlabel("Tempo (s)", color=cor_texto)
    ax_emg.legend(facecolor=cor_fundo, labelcolor=cor_texto)
    ax_emg.grid(True, linestyle='--', alpha=0.3)
    ax_emg.tick_params(colors=cor_texto); 
    ax_emg.xaxis.label.set_color(cor_texto); 
    ax_emg.yaxis.label.set_color(cor_texto)

    plt.tight_layout()
    return fig

def gerar_grafico_evolucao(df_hist, cor_fundo, cor_texto, cor_eixos):
    """
    Gera gráfico de evolução com faixas normativas (Bands).
    """
    # Prepara dados
    df_hist['DATA_OBJ'] = pd.to_datetime(df_hist['DATA_HORA'])
    df_hist = df_hist.sort_values('DATA_OBJ')
    datas = df_hist['DATA_OBJ'].dt.strftime('%d/%m %H:%M')
    
    # Cria figura
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), facecolor=cor_fundo)
    fig.patch.set_facecolor(cor_fundo)
    fig.suptitle(f"Evolução Clínica e Faixas de Normalidade", fontsize=22, weight='bold', color=cor_texto)
    
    # Função auxiliar de estilo
    def sty(ax, t, yl):
        ax.set_facecolor(cor_fundo)
        ax.set_title(t, color=cor_texto, fontsize=14, weight='bold')
        ax.set_ylabel(yl, color=cor_texto, fontsize=12)
        ax.tick_params(colors=cor_texto, rotation=45)
        # Grid mais leve para não brigar com as faixas
        ax.grid(True, alpha=0.15, linestyle=':', color=cor_texto)
        for s in ax.spines.values(): s.set_color(cor_texto)

    # --- 1. GRÁFICO DE CADÊNCIA ---
    ax = axs[0]
    # Faixas Normativas (Winter, 1991)
    ax.axhspan(100, 120, color='green', alpha=0.15, label='Normal (100-120)') # Zona Ideal
    ax.axhspan(80, 100, color='yellow', alpha=0.10, label='Funcional (80-100)') # Zona Funcional
    ax.axhspan(60, 80, color='orange', alpha=0.10, label='Atenção (60-80)')
    ax.axhspan(0, 60, color='red', alpha=0.10, label='Risco (<60)') # Zona de Risco
    
    ax.plot(datas, df_hist['CADENCIA'], 'o-', color='#00FFC8', lw=2.5, label='Paciente')
    sty(ax, "Cadência (Ritmo)", "Passos/min")
    ax.legend(facecolor=cor_fundo, labelcolor=cor_texto, loc='upper left', fontsize='small')

    # --- 2. GRÁFICO DE APOIO (% DO CICLO) ---
    ax = axs[1]
    # Faixas Normativas (Perry, 2010) - 60% a 62% é o ouro, aceita-se 58-64%
    ax.axhspan(52, 65, color='green', alpha=0.15, label='Normal/Funcional (52-65%)') 
    ax.axhspan(65, 72, color='yellow', alpha=0.10, label='Atenção') 
    ax.axhspan(45, 52, color='yellow', alpha=0.10)
    
    val_apoio = df_hist['PCT_APOIO']
    ax.plot(datas, val_apoio, 's-', color='#0056B3', lw=2.5, label='% Apoio')
    
    # Linha de Ouro (60%)
    ax.axhline(y=60, color=cor_texto, linestyle='--', alpha=0.5, lw=1)
    
    sty(ax, "% do Ciclo em Apoio", "% do Ciclo")
    ax.legend(facecolor=cor_fundo, labelcolor=cor_texto, loc='upper left', fontsize='small')
    
    # --- 3. GRÁFICO DE RMS (EFICIÊNCIA) ---
    ax = axs[2]
   
    ax.plot(datas, df_hist['RMS'], 'o-', color='#D500F9', lw=2.5, label='RMS Paciente')
    sty(ax, "Ativação Média (RMS)", "Amplitude (0-1)")
    ax.legend(facecolor=cor_fundo, labelcolor=cor_texto, loc='upper left', fontsize='small')
    
    plt.tight_layout()
    return fig
