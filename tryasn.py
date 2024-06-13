import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns

# Sidebar option menu
with st.sidebar:
    Pilihan = option_menu('ASN', ['EEG1', 'EEG2', 'ECG'], icons=['house', 'basket2', 'card-list'], default_index=0)

    # Session state for buttons
    if 'button1_clicked' not in st.session_state:
        st.session_state.button1_clicked = False
    def on_button1_click():
        st.session_state.button1_clicked = True

    if 'button2_clicked' not in st.session_state:
        st.session_state.button2_clicked = False
    def on_button2_click():
        st.session_state.button2_clicked = True

    if 'button3_clicked' not in st.session_state:
        st.session_state.button3_clicked = False
    def on_button3_click():
        st.session_state.button3_clicked = True

    if 'button4_clicked' not in st.session_state:
        st.session_state.button4_clicked = False
    def on_button4_click():
        st.session_state.button4_clicked = True
    
    if 'button5_clicked' not in st.session_state:
        st.session_state.button5_clicked = False
    def on_button5_click():
        st.session_state.button5_clicked = True

    if 'button6_clicked' not in st.session_state:
        st.session_state.button6_clicked = False
    def on_button6_click():
        st.session_state.button6_clicked = True
    
    if 'button7_clicked' not in st.session_state:
        st.session_state.button7_clicked = False
    def on_button7_click():
        st.session_state.button7_clicked = True

    if 'button8_clicked' not in st.session_state:
        st.session_state.button8_clicked = False
    def on_button8_click():
        st.session_state.button8_clicked = True

    if 'button9_clicked' not in st.session_state:
        st.session_state.button9_clicked = False
    def on_button9_click():
        st.session_state.button9_clicked = True

    if 'button10_clicked' not in st.session_state:
        st.session_state.button10_clicked = False
    def on_button10_click():
        st.session_state.button10_clicked = True

    if 'button11_clicked' not in st.session_state:
        st.session_state.button11_clicked = False
    def on_button11_click():
        st.session_state.button11_clicked = True

    if 'button12_clicked' not in st.session_state:
        st.session_state.button12_clicked = False
    def on_button12_click():
        st.session_state.button12_clicked = True
    
    if 'button13_clicked' not in st.session_state:
        st.session_state.button13_clicked = False
    def on_button13_click():
        st.session_state.button13_clicked = True


    if 'button14_clicked' not in st.session_state:
        st.session_state.button14_clicked = False
    def on_button14_click():
        st.session_state.button14_clicked = True

    if 'button15_clicked' not in st.session_state:
        st.session_state.button15_clicked = False
    def on_button15_click():
        st.session_state.button15_clicked = True

    if 'button16_clicked' not in st.session_state:
        st.session_state.button16_clicked = False
    def on_button16_click():
        st.session_state.button16_clicked = True
    
    if 'button17_clicked' not in st.session_state:
        st.session_state.button17_clicked = False
    def on_button17_click():
        st.session_state.button17_clicked = True

    if 'button18_clicked' not in st.session_state:
        st.session_state.button18_clicked = False
    def on_button18_click():
        st.session_state.button18_clicked = True

    if 'button19_clicked' not in st.session_state:
        st.session_state.button19_clicked = False
    def on_button19_click():
        st.session_state.button19_clicked = True
        
    if 'button20_clicked' not in st.session_state:
        st.session_state.button20_clicked = False
    def on_button20_click():
        st.session_state.button20_clicked = True



@st.cache_data
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file, header=None)
    df.columns = ['huruf', 'nilai']
    return df
    
@st.cache_data
def filter_data(df):
    data_a = df[df['huruf'] == 'a']['nilai'].reset_index(drop=True)
    data_b = df[df['huruf'] == 'b']['nilai'].reset_index(drop=True)
    data_c = df[df['huruf'] == 'c']['nilai'].reset_index(drop=True)
    return data_a, data_b, data_c

@st.cache_data()
def fourier_transform(signal):
    N = len(signal)
    X_real = np.zeros(N)
    X_imaj = np.zeros(N)
    fft_result = np.zeros(N)
    
    for k in range(N):
        for n in range(N):
            X_real[k] += signal[n] * np.cos(2 * np.pi * k * n / N)
            X_imaj[k] -= signal[n] * np.sin(2 * np.pi * k * n / N)
        fft_result[k] = np.sqrt(X_real[k]**2 + X_imaj[k]**2)
    return fft_result

@st.cache_data()
def calculate_frequency(N, sampling_rate):
    return np.arange(N) * sampling_rate / N


if Pilihan == 'EEG1':
    st.header('Masukkan Data')
    uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=None)
        df.columns = ['huruf', 'nilai']
        data_a,data_b, data_c = filter_data(df)
        
        max_y_value = 6e6
        data_b = np.clip(data_b, 0, max_y_value)
        data_c = np.clip(data_c, 0, max_y_value)
        
        jumlah_fp1 = len(data_b)
        jumlah_fp2 = len(data_c)
        fs = 125
        
        elapsed_time_fp1 = data_b.index * (1 / fs)
        elapsed_time_fp2 = data_c.index * (1 / fs)
        
        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Scatter(x=elapsed_time_fp1, y=data_b, mode='lines', name='EEG (fp1)', line=dict(color='blue')), row=1, col=1)
        fig.update_xaxes(title_text="Elapsed Time", row=1, col=1)
        fig.update_yaxes(title_text="Nilai", range=[0, 6e6], row=1, col=1)
        fig.add_trace(go.Scatter(x=elapsed_time_fp2, y=data_c, mode='lines', name='EEG (fp2)', line=dict(color='red')), row=2, col=1)
        fig.update_xaxes(title_text="Elapsed Time", row=2, col=1)
        fig.update_yaxes(title_text="Nilai", range=[0, 6e6], row=2, col=1)
        fig.update_layout(height=600, width=1500, title_text="Plot Data EEG (fp1 & fp2)")
    
        
        but4 = st.button('Sinyal awal', on_click=on_button4_click)
        if st.session_state.button4_clicked:
            st.plotly_chart(fig)
        
        st.header('Bandpass Filter')
        fc_lpf = st.number_input("FC LPF", 0)
        lpf_eeg2 = np.zeros(jumlah_fp1)
        
        T = 1 / fs
        w = 2 * math.pi * fc_lpf
        a0 = w**2
        a1 = 2 * (w**2)
        b1 = ((8 / (T**2)) - (2 * (w**2)))
        c0 = ((4 / (T**2)) - ((2 * (math.sqrt(2)) * w) / T) + (w**2))
        c1 = ((4 / (T**2)) + ((2 * (math.sqrt(2)) * w) / T) + (w**2))
        
        for n in range(2, jumlah_fp1):
            lpf_eeg2[n] = ((b1 * lpf_eeg2[n - 1]) - (c0 * lpf_eeg2[n - 2]) + (a0 * data_b[n]) + (a1 * data_b[n - 1]) + (a0 * data_b[n - 2])) / c1
        
        elapsed_time = np.arange(len(lpf_eeg2)) * (1 / fs)
        
        fc_hpf = st.number_input("FC HPF", 0)
        hpf_eeg2 = np.zeros(np.size(lpf_eeg2))
        w = 2 * math.pi * fc_hpf
        e0 = 4 * T
        e1 = 8 * T
        e2 = 4 * T
        d0 = ((2 * (w**2) * (T**2)) - 8)
        d1 = (((w**2) * (T**2)) - (2 * (math.sqrt(2)) * T * w) + 4)
        d2 = ((w**2) * (T**2)) + (2 * (math.sqrt(2)) * T * w) + 4
        
        for n in range(2, jumlah_fp1):
            hpf_eeg2[n] = ((e0 * lpf_eeg2[n]) - (e1 * lpf_eeg2[n - 1]) + (e2 * lpf_eeg2[n - 2]) - (d0 * hpf_eeg2[n - 1]) - (d1 * hpf_eeg2[n - 2])) / d2
        
        elapsed_time = np.arange(len(hpf_eeg2)) * (1 / fs)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=elapsed_time, y=hpf_eeg2, mode='lines', name='BPF EEG (fp1)', line=dict(color='blue')))
        fig1.update_layout(height=500, width=1500, title="Plot Data BPF EEG (fp1)", xaxis_title="Elapsed Time", yaxis_title="Nilai")
        
        but6= st.button('Sinyal BPF', on_click=on_button6_click)
        if st.session_state.button6_clicked:
            st.plotly_chart(fig1)


        st.header('Welchs Method')
        ptp2 = 0
        waktu2 = np.zeros(np.size(hpf_eeg2))
        selisih2 = np.zeros(np.size(hpf_eeg2))
        for n in range(np.size(hpf_eeg2) - 1):
            if hpf_eeg2[n] < hpf_eeg2[n + 1]:
                waktu2[ptp2] = n / fs
                selisih2[ptp2] = waktu2[ptp2] - waktu2[ptp2 - 1]
                ptp2 += 1
        ptp2 -= 1
        
        n= np.arange(0, ptp2, 1, dtype=int)
        n_subset = n[0:5000]  # Subset 0-5000 data
        bpf_eeg_subset = hpf_eeg2[0:5000]
        
       
        
        M = len(bpf_eeg_subset) - 1
        hamming_window = np.zeros(M + 1)
        for i in range(M + 1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
            
        bpf_eeg_subset_windowed = bpf_eeg_subset * hamming_window
        
            
        fft_result = fourier_transform(bpf_eeg_subset_windowed)
        sampling_rate = 125
        fft_freq = calculate_frequency(len(bpf_eeg_subset_windowed), sampling_rate)
        half_point = len(fft_freq) // 2
        fft_freq_half = fft_freq[:half_point]
        fft_result_half = fft_result[:half_point]

        st.subheader('Subset 0-5000')

        fig2 = go.Figure(data=go.Scatter(x=n_subset, y=bpf_eeg_subset_windowed, mode='lines'))
        fig2.update_layout(height=400,width=1500,title="Subset 0-4999 with Hamming Window",xaxis_title="n",yaxis_title="Nilai",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))

        
        fig_fft1 = go.Figure(data=go.Scatter(x=fft_freq_half, y=np.abs(fft_result_half),mode="lines"))
        fig_fft1.update_layout( height=400, width=1500, title="FFT Subset", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", xaxis=dict(showline=True, showgrid=True), yaxis=dict(showline=True, showgrid=True))
        but5 = st.button('Cek', on_click=on_button5_click)
        if st.session_state.button5_clicked:
            st.plotly_chart(fig2)
            st.plotly_chart(fig_fft1)
      


        st.subheader('Subset 2500-7500')
        n_subset1 = n[2500:7500] 
        bpf_eeg_subset1 = hpf_eeg2[2500:7500]

        M = len(bpf_eeg_subset1) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed1 = bpf_eeg_subset1 * hamming_window
        
        fft_result1 = fourier_transform(bpf_eeg_subset_windowed1)
        sampling_rate = 125
        fft_freq1 = calculate_frequency(len(bpf_eeg_subset_windowed1), sampling_rate)
        half_point1 = len(fft_freq1) // 2
        fft_freq_half1 = fft_freq1[:half_point1]
        fft_result_half1 = fft_result1[:half_point1]

        fig3 = go.Figure(data=go.Scatter(x=n_subset1, y=bpf_eeg_subset_windowed1, mode='lines'))
        fig3.update_layout(height=400, width=1500, title="Subset 2500-7500 with Hamming Window",xaxis_title="n",yaxis_title="Nilai",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft2= go.Figure(data=go.Scatter(x=fft_freq_half1, y=np.abs(fft_result_half1),mode="lines"))
        fig_fft2.update_layout( height=400, width=1500,title="FFT of Subset",xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but6= st.button('Cek ', on_click=on_button6_click)
        if st.session_state.button6_clicked:
            st.plotly_chart(fig3)
            st.plotly_chart(fig_fft2)


        
        st.subheader('Subset 5000-10000')
        n_subset2 = n[5000:10000]
        bpf_eeg_subset2 = hpf_eeg2[5000:10000]

        M = len(bpf_eeg_subset2) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed2 = bpf_eeg_subset2 * hamming_window
        
        fft_result2 = fourier_transform(bpf_eeg_subset_windowed2)
        sampling_rate = 125
        fft_freq2 = calculate_frequency(len(bpf_eeg_subset_windowed2), sampling_rate)
        half_point2 = len(fft_freq2) // 2
        fft_freq_half2 = fft_freq2[:half_point2]
        fft_result_half2 = fft_result2[:half_point2]

        fig4 = go.Figure(data=go.Scatter(x=n_subset2, y=bpf_eeg_subset_windowed2, mode='lines'))
        fig4.update_layout(height=400,width=1500,title="Subset 5000-10000 with Hamming Window",xaxis_title="n",yaxis_title="Nilai",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft3 = go.Figure(data=go.Scatter(x=fft_freq_half2, y=np.abs(fft_result_half2),mode="lines"))
        fig_fft3.update_layout(height=400,width=1500,title="FFT of Subset",xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but14= st.button('Cek  ', on_click=on_button14_click)
        if st.session_state.button14_clicked:
            st.plotly_chart(fig4)
            st.plotly_chart(fig_fft3)



        st.subheader('Subset 7500-12500')
        n_subset3 = n[7500:12500] #ambil subset 0-100 data
        bpf_eeg_subset3 = hpf_eeg2[7500:12500]

        M = len(bpf_eeg_subset3) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed3 = bpf_eeg_subset3 * hamming_window

        fft_result3 = fourier_transform(bpf_eeg_subset_windowed3)
        sampling_rate = 125
        fft_freq3 = calculate_frequency(len(bpf_eeg_subset_windowed3), sampling_rate)
        half_point3 = len(fft_freq3) // 2
        fft_freq_half3 = fft_freq3[:half_point3]
        fft_result_half3 = fft_result3[:half_point3]
        
        fig5 = go.Figure(data=go.Scatter(x=n_subset3, y=bpf_eeg_subset_windowed3, mode='lines'))
        fig5.update_layout( height=400, width=1500, title="Subset 7500-12500 with Hamming Window", xaxis_title="n", yaxis_title="Nilai", xaxis=dict(showline=True, showgrid=True), yaxis=dict(showline=True, showgrid=True))
        
        fig_fft4 = go.Figure(data=go.Scatter(x=fft_freq_half3, y=np.abs(fft_result_half3),mode="lines"))
        fig_fft4.update_layout(height=400,width=1500,title="FFT of Subset",xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but9= st.button('Cek   ', on_click=on_button9_click)
        if st.session_state.button9_clicked:
            st.plotly_chart(fig5)
            st.plotly_chart(fig_fft4)


        
        st.subheader('Subset 10000-15000')
        n_subset4 = n[10000:15000] #ambil subset 0-100 data
        bpf_eeg_subset4 = hpf_eeg2[10000:15000]


        M = len(bpf_eeg_subset4) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed4 = bpf_eeg_subset4 * hamming_window
            
        fft_result4 = fourier_transform(bpf_eeg_subset_windowed4)
        sampling_rate = 125
        fft_freq4 = calculate_frequency(len(bpf_eeg_subset_windowed4), sampling_rate)
        half_point4 = len(fft_freq4) // 2
        fft_freq_half4 = fft_freq4[:half_point4]
        fft_result_half4 = fft_result4[:half_point4]

        fig6= go.Figure(data=go.Scatter(x=n_subset4, y=bpf_eeg_subset_windowed4, mode='lines'))
        fig6.update_layout( height=400, width=1500, title="Subset 10000-15000 with Hamming Window", xaxis_title="n", yaxis_title="Nilai", xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft5= go.Figure(data=go.Scatter(x=fft_freq_half4, y=np.abs(fft_result_half4),mode="lines"))
        fig_fft5.update_layout( height=400, width=1500, title="FFT of Subset", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but10= st.button('Cek    ', on_click=on_button10_click)
        if st.session_state.button10_clicked:
            st.plotly_chart(fig6)
            st.plotly_chart(fig_fft5)




        st.subheader('Subset 12500-17500')
        n_subset5 = n[12500:17500]
        bpf_eeg_subset5 = hpf_eeg2[12500:17500]

        M = len(bpf_eeg_subset5) - 1
        hamming_window = np.zeros(M+1)
        
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed5 = bpf_eeg_subset5 * hamming_window
            
        
        fft_result5 = fourier_transform(bpf_eeg_subset_windowed5)
        sampling_rate = 125
        fft_freq5 = calculate_frequency(len(bpf_eeg_subset_windowed5), sampling_rate)
        half_point5 = len(fft_freq5) // 2
        fft_freq_half5 = fft_freq5[:half_point5]
        fft_result_half5 = fft_result5[:half_point5]
        
        fig7= go.Figure(data=go.Scatter(x=n_subset5, y=bpf_eeg_subset_windowed5, mode='lines'))
        fig7.update_layout(height=400,width=1500,title="Subset 12500-17500 with Hamming Window",xaxis_title="n",yaxis_title="Nilai",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft6 = go.Figure(data=go.Scatter(x=fft_freq_half5, y=np.abs(fft_result_half5),mode="lines"))
        fig_fft6.update_layout(height=400,width=1500,title="FFT of Subset",xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but11= st.button('Cek     ', on_click=on_button11_click)
        if st.session_state.button11_clicked:
            st.plotly_chart(fig7)
            st.plotly_chart(fig_fft6)



        st.subheader('Subset 15000-20000')
        n_subset6 = n[15000:20000]
        bpf_eeg_subset6 = hpf_eeg2[15000:20000]
        
        M = len(bpf_eeg_subset6) - 1
        hamming_window = np.zeros(M+1)
        
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed6 = bpf_eeg_subset6 * hamming_window
        
            
        fft_result6 = fourier_transform(bpf_eeg_subset_windowed6)
        sampling_rate = 125
        fft_freq6 = calculate_frequency(len(bpf_eeg_subset_windowed6), sampling_rate)
        half_point6 = len(fft_freq6) // 2
        fft_freq_half6 = fft_freq6[:half_point6]
        fft_result_half6 = fft_result6[:half_point6]

        fig8 = go.Figure(data=go.Scatter(x=n_subset6, y=bpf_eeg_subset_windowed6, mode='lines'))
        fig8.update_layout(height=400,width=1500,title="Subset 15000-20000 with Hamming Window",xaxis_title="n",yaxis_title="Nilai",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft7= go.Figure(data=go.Scatter(x=fft_freq_half6, y=np.abs(fft_result_half6),mode="lines"))
        fig_fft7.update_layout( height=400, width=1500, title="FFT of Subset", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but12= st.button('Cek      ', on_click=on_button12_click)
        if st.session_state.button12_clicked:
            st.plotly_chart(fig8)
            st.plotly_chart(fig_fft7)




        st.subheader('Subset 17500-22500')
        n_subset7 = n[17500:22500]
        bpf_eeg_subset7 = hpf_eeg2[17500:22500]

        M = len(bpf_eeg_subset7) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed7 = bpf_eeg_subset7 * hamming_window

        
        fft_result7 = fourier_transform(bpf_eeg_subset_windowed7)
        sampling_rate = 125
        fft_freq7 = calculate_frequency(len(bpf_eeg_subset_windowed7), sampling_rate)
        half_point7 = len(fft_freq7) // 2
        fft_freq_half7 = fft_freq7[:half_point7]
        fft_result_half7 = fft_result7[:half_point7]

        fig9= go.Figure(data=go.Scatter(x=n_subset7, y=bpf_eeg_subset_windowed7, mode='lines'))
        fig9.update_layout(height=400,width=1500,title="Subset 17500-225000 with Hamming Window",xaxis_title="n",yaxis_title="Nilai",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft8 = go.Figure(data=go.Scatter(x=fft_freq_half7, y=np.abs(fft_result_half7),mode="lines"))
        fig_fft8.update_layout( height=400,width=1500,title="FFT of Subset",xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but13= st.button('Cek       ', on_click=on_button13_click)
        if st.session_state.button13_clicked:
            st.plotly_chart(fig9)
            st.plotly_chart(fig_fft8)


        st.subheader('FFT Total')

        average_fft = (fft_result + fft_result1 +  fft_result2 + fft_result3 + fft_result4 + fft_result5 + fft_result6 + fft_result7) / 8
        average_freq = (fft_freq + fft_freq1 + fft_freq2 +  fft_freq3 +  fft_freq4 +  fft_freq5 +  fft_freq6 +  fft_freq7)/ 8
        half_point_total = len(average_freq) // 2
        fft_freq_total = average_freq[:half_point_total]
        fft_result_total = average_fft[:half_point_total]
        
        fig_fft = go.Figure(data=go.Scatter(x=fft_freq_total, y=np.abs(fft_result_total), mode='lines'))
        fig_fft.update_layout(
            title="FFT TOTAL",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True)
        )
        but7 = st.button('Lihat FFT', on_click=on_button7_click)
        if st.session_state.button7_clicked:
            st.plotly_chart(fig_fft)
       
        def calculate_mean_power_frequency(freq, psd):
            freq = np.asarray(freq)
            psd = np.asarray(psd)
            mpf = np.sum(freq * psd) / np.sum(psd)
            return mpf
        
        fft_freq_total = fft_freq_half 
        fft_result_total = np.abs(fft_result_half)
        
        mpf = calculate_mean_power_frequency(fft_freq_total, fft_result_total)
        alpha_range = (8, 13)
        beta_range = (13, 30)
        gamma_range = (30, 40)
        def determine_band(mpf):
            if alpha_range[0] <= mpf < alpha_range[1]:
                return 'Alpha'
            elif beta_range[0] <= mpf < beta_range[1]:
                return 'Beta'
            elif gamma_range[0] <= mpf < gamma_range[1]:
                return 'Gamma'
            else:
                return 'Out of defined ranges'
        band = determine_band(mpf)
        but8=st.button('Hasil', on_click=on_button8_click)
        if st.session_state.button8_clicked:
            st.success(f'Mean frequency Power (MPF): {mpf}')
            st.success(f'Rentang MPF: {band}')

if Pilihan == 'EEG2':
    st.header('Masukkan Data')
    uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=None)
        df.columns = ['huruf', 'nilai']
        data_a,data_b, data_c = filter_data(df)
        
        max_y_value = 6e6
        data_b = np.clip(data_b, 0, max_y_value)
        data_c = np.clip(data_c, 0, max_y_value)
        
        jumlah_fp1 = len(data_b)
        jumlah_fp2 = len(data_c)
        fs = 125
        
        elapsed_time_fp1 = data_b.index * (1 / fs)
        elapsed_time_fp2 = data_c.index * (1 / fs)
        
        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Scatter(x=elapsed_time_fp1, y=data_b, mode='lines', name='EEG (fp1)', line=dict(color='blue')), row=1, col=1)
        fig.update_xaxes(title_text="Elapsed Time", row=1, col=1)
        fig.update_yaxes(title_text="Nilai", range=[0, 6e6], row=1, col=1)
        fig.add_trace(go.Scatter(x=elapsed_time_fp2, y=data_c, mode='lines', name='EEG (fp2)', line=dict(color='red')), row=2, col=1)
        fig.update_xaxes(title_text="Elapsed Time", row=2, col=1)
        fig.update_yaxes(title_text="Nilai", range=[0, 6e6], row=2, col=1)
        fig.update_layout(height=600, width=1500, title_text="Plot Data EEG (fp1 & fp2)")
    
        
        but4 = st.button('Sinyal awal', on_click=on_button4_click)
        if st.session_state.button4_clicked:
            st.plotly_chart(fig)
        
        st.header('Bandpass Filter')
        fc_lpf = st.number_input("FC LPF", 0)
        lpf_eeg2 = np.zeros(jumlah_fp2)
        
        T = 1 / fs
        w = 2 * math.pi * fc_lpf
        a0 = w**2
        a1 = 2 * (w**2)
        b1 = ((8 / (T**2)) - (2 * (w**2)))
        c0 = ((4 / (T**2)) - ((2 * (math.sqrt(2)) * w) / T) + (w**2))
        c1 = ((4 / (T**2)) + ((2 * (math.sqrt(2)) * w) / T) + (w**2))
        
        for n in range(2, jumlah_fp2):
            lpf_eeg2[n] = ((b1 * lpf_eeg2[n - 1]) - (c0 * lpf_eeg2[n - 2]) + (a0 * data_c[n]) + (a1 * data_c[n - 1]) + (a0 * data_c[n - 2])) / c1
        
        elapsed_time = np.arange(len(lpf_eeg2)) * (1 / fs)
        
        fc_hpf = st.number_input("FC HPF", 0)
        hpf_eeg2 = np.zeros(np.size(lpf_eeg2))
        w = 2 * math.pi * fc_hpf
        e0 = 4 * T
        e1 = 8 * T
        e2 = 4 * T
        d0 = ((2 * (w**2) * (T**2)) - 8)
        d1 = (((w**2) * (T**2)) - (2 * (math.sqrt(2)) * T * w) + 4)
        d2 = ((w**2) * (T**2)) + (2 * (math.sqrt(2)) * T * w) + 4
        
        for n in range(2, jumlah_fp2):
            hpf_eeg2[n] = ((e0 * lpf_eeg2[n]) - (e1 * lpf_eeg2[n - 1]) + (e2 * lpf_eeg2[n - 2]) - (d0 * hpf_eeg2[n - 1]) - (d1 * hpf_eeg2[n - 2])) / d2
        
        elapsed_time = np.arange(len(hpf_eeg2)) * (1 / fs)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=elapsed_time, y=hpf_eeg2, mode='lines', name='BPF EEG (fp1)', line=dict(color='blue')))
        fig1.update_layout(height=500, width=1500, title="Plot Data BPF EEG (fp1)", xaxis_title="Elapsed Time", yaxis_title="Nilai")
        
        but6= st.button('Sinyal HPF', on_click=on_button6_click)
        if st.session_state.button6_clicked:
            st.plotly_chart(fig1)
        
        st.header('Welchs Method')
        ptp2 = 0
        waktu2 = np.zeros(np.size(hpf_eeg2))
        selisih2 = np.zeros(np.size(hpf_eeg2))
        for n in range(np.size(hpf_eeg2) - 1):
            if hpf_eeg2[n] < hpf_eeg2[n + 1]:
                waktu2[ptp2] = n / fs
                selisih2[ptp2] = waktu2[ptp2] - waktu2[ptp2 - 1]
                ptp2 += 1
        ptp2 -= 1
        
        n= np.arange(0, ptp2, 1, dtype=int)
        n_subset = n[0:5000]  # Subset 0-5000 data
        bpf_eeg_subset = hpf_eeg2[0:5000]
        
       
        
        M = len(bpf_eeg_subset) - 1
        hamming_window = np.zeros(M + 1)
        for i in range(M + 1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
            
        bpf_eeg_subset_windowed = bpf_eeg_subset * hamming_window
        
            
        fft_result = fourier_transform(bpf_eeg_subset_windowed)
        sampling_rate = 125
        fft_freq = calculate_frequency(len(bpf_eeg_subset_windowed), sampling_rate)
        half_point = len(fft_freq) // 2
        fft_freq_half = fft_freq[:half_point]
        fft_result_half = fft_result[:half_point]

        st.subheader('Subset 0-5000')

        fig2 = go.Figure(data=go.Scatter(x=n_subset, y=bpf_eeg_subset_windowed, mode='lines'))
        fig2.update_layout(height=400,width=1500,title="Subset 0-4999 with Hamming Window",xaxis_title="n",yaxis_title="Nilai",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))

        
        fig_fft1 = go.Figure(data=go.Scatter(x=fft_freq_half, y=np.abs(fft_result_half),mode="lines"))
        fig_fft1.update_layout( height=400, width=1500, title="FFT Subset", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", xaxis=dict(showline=True, showgrid=True), yaxis=dict(showline=True, showgrid=True))
        but5 = st.button('Cek', on_click=on_button5_click)
        if st.session_state.button5_clicked:
            st.plotly_chart(fig2)
            st.plotly_chart(fig_fft1)
      


        st.subheader('Subset 2500-7500')
        n_subset1 = n[2500:7500] 
        bpf_eeg_subset1 = hpf_eeg2[2500:7500]

        M = len(bpf_eeg_subset1) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed1 = bpf_eeg_subset1 * hamming_window
        
        fft_result1 = fourier_transform(bpf_eeg_subset_windowed1)
        sampling_rate = 125
        fft_freq1 = calculate_frequency(len(bpf_eeg_subset_windowed1), sampling_rate)
        half_point1 = len(fft_freq1) // 2
        fft_freq_half1 = fft_freq1[:half_point1]
        fft_result_half1 = fft_result1[:half_point1]

        fig3 = go.Figure(data=go.Scatter(x=n_subset1, y=bpf_eeg_subset_windowed1, mode='lines'))
        fig3.update_layout(height=400, width=1500, title="Subset 2500-7500 with Hamming Window",xaxis_title="n",yaxis_title="Nilai",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft2= go.Figure(data=go.Scatter(x=fft_freq_half1, y=np.abs(fft_result_half1),mode="lines"))
        fig_fft2.update_layout( height=400, width=1500,title="FFT of Subset",xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but6= st.button('Cek ', on_click=on_button6_click)
        if st.session_state.button6_clicked:
            st.plotly_chart(fig3)
            st.plotly_chart(fig_fft2)


        
        st.subheader('Subset 5000-10000')
        n_subset2 = n[5000:10000]
        bpf_eeg_subset2 = hpf_eeg2[5000:10000]

        M = len(bpf_eeg_subset2) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed2 = bpf_eeg_subset2 * hamming_window
        
        fft_result2 = fourier_transform(bpf_eeg_subset_windowed2)
        sampling_rate = 125
        fft_freq2 = calculate_frequency(len(bpf_eeg_subset_windowed2), sampling_rate)
        half_point2 = len(fft_freq2) // 2
        fft_freq_half2 = fft_freq2[:half_point2]
        fft_result_half2 = fft_result2[:half_point2]

        fig4 = go.Figure(data=go.Scatter(x=n_subset2, y=bpf_eeg_subset_windowed2, mode='lines'))
        fig4.update_layout(height=400,width=1500,title="Subset 5000-10000 with Hamming Window",xaxis_title="n",yaxis_title="Nilai",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft3 = go.Figure(data=go.Scatter(x=fft_freq_half2, y=np.abs(fft_result_half2),mode="lines"))
        fig_fft3.update_layout(height=400,width=1500,title="FFT of Subset",xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but14= st.button('Cek  ', on_click=on_button14_click)
        if st.session_state.button14_clicked:
            st.plotly_chart(fig4)
            st.plotly_chart(fig_fft3)



        st.subheader('Subset 7500-12500')
        n_subset3 = n[7500:12500] #ambil subset 0-100 data
        bpf_eeg_subset3 = hpf_eeg2[7500:12500]

        M = len(bpf_eeg_subset3) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed3 = bpf_eeg_subset3 * hamming_window

        fft_result3 = fourier_transform(bpf_eeg_subset_windowed3)
        sampling_rate = 125
        fft_freq3 = calculate_frequency(len(bpf_eeg_subset_windowed3), sampling_rate)
        half_point3 = len(fft_freq3) // 2
        fft_freq_half3 = fft_freq3[:half_point3]
        fft_result_half3 = fft_result3[:half_point3]
        
        fig5 = go.Figure(data=go.Scatter(x=n_subset3, y=bpf_eeg_subset_windowed3, mode='lines'))
        fig5.update_layout( height=400, width=1500, title="Subset 7500-12500 with Hamming Window", xaxis_title="n", yaxis_title="Nilai", xaxis=dict(showline=True, showgrid=True), yaxis=dict(showline=True, showgrid=True))
        
        fig_fft4 = go.Figure(data=go.Scatter(x=fft_freq_half3, y=np.abs(fft_result_half3),mode="lines"))
        fig_fft4.update_layout(height=400,width=1500,title="FFT of Subset",xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but9= st.button('Cek   ', on_click=on_button9_click)
        if st.session_state.button9_clicked:
            st.plotly_chart(fig5)
            st.plotly_chart(fig_fft4)


        
        st.subheader('Subset 10000-15000')
        n_subset4 = n[10000:15000] #ambil subset 0-100 data
        bpf_eeg_subset4 = hpf_eeg2[10000:15000]


        M = len(bpf_eeg_subset4) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed4 = bpf_eeg_subset4 * hamming_window
            
        fft_result4 = fourier_transform(bpf_eeg_subset_windowed4)
        sampling_rate = 125
        fft_freq4 = calculate_frequency(len(bpf_eeg_subset_windowed4), sampling_rate)
        half_point4 = len(fft_freq4) // 2
        fft_freq_half4 = fft_freq4[:half_point4]
        fft_result_half4 = fft_result4[:half_point4]

        fig6= go.Figure(data=go.Scatter(x=n_subset4, y=bpf_eeg_subset_windowed4, mode='lines'))
        fig6.update_layout( height=400, width=1500, title="Subset 10000-15000 with Hamming Window", xaxis_title="n", yaxis_title="Nilai", xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft5= go.Figure(data=go.Scatter(x=fft_freq_half4, y=np.abs(fft_result_half4),mode="lines"))
        fig_fft5.update_layout( height=400, width=1500, title="FFT of Subset", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but10= st.button('Cek    ', on_click=on_button10_click)
        if st.session_state.button10_clicked:
            st.plotly_chart(fig6)
            st.plotly_chart(fig_fft5)




        st.subheader('Subset 12500-17500')
        n_subset5 = n[12500:17500]
        bpf_eeg_subset5 = hpf_eeg2[12500:17500]

        M = len(bpf_eeg_subset5) - 1
        hamming_window = np.zeros(M+1)
        
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed5 = bpf_eeg_subset5 * hamming_window
            
        
        fft_result5 = fourier_transform(bpf_eeg_subset_windowed5)
        sampling_rate = 125
        fft_freq5 = calculate_frequency(len(bpf_eeg_subset_windowed5), sampling_rate)
        half_point5 = len(fft_freq5) // 2
        fft_freq_half5 = fft_freq5[:half_point5]
        fft_result_half5 = fft_result5[:half_point5]
        
        fig7= go.Figure(data=go.Scatter(x=n_subset5, y=bpf_eeg_subset_windowed5, mode='lines'))
        fig7.update_layout(height=400,width=1500,title="Subset 12500-17500 with Hamming Window",xaxis_title="n",yaxis_title="Nilai",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft6 = go.Figure(data=go.Scatter(x=fft_freq_half5, y=np.abs(fft_result_half5),mode="lines"))
        fig_fft6.update_layout(height=400,width=1500,title="FFT of Subset",xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but11= st.button('Cek     ', on_click=on_button11_click)
        if st.session_state.button11_clicked:
            st.plotly_chart(fig7)
            st.plotly_chart(fig_fft6)



        st.subheader('Subset 15000-20000')
        n_subset6 = n[15000:20000]
        bpf_eeg_subset6 = hpf_eeg2[15000:20000]
        
        M = len(bpf_eeg_subset6) - 1
        hamming_window = np.zeros(M+1)
        
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed6 = bpf_eeg_subset6 * hamming_window
        
            
        fft_result6 = fourier_transform(bpf_eeg_subset_windowed6)
        sampling_rate = 125
        fft_freq6 = calculate_frequency(len(bpf_eeg_subset_windowed6), sampling_rate)
        half_point6 = len(fft_freq6) // 2
        fft_freq_half6 = fft_freq6[:half_point6]
        fft_result_half6 = fft_result6[:half_point6]

        fig8 = go.Figure(data=go.Scatter(x=n_subset6, y=bpf_eeg_subset_windowed6, mode='lines'))
        fig8.update_layout(height=400,width=1500,title="Subset 15000-20000 with Hamming Window",xaxis_title="n",yaxis_title="Nilai",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft7= go.Figure(data=go.Scatter(x=fft_freq_half6, y=np.abs(fft_result_half6),mode="lines"))
        fig_fft7.update_layout( height=400, width=1500, title="FFT of Subset", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but12= st.button('Cek      ', on_click=on_button12_click)
        if st.session_state.button12_clicked:
            st.plotly_chart(fig8)
            st.plotly_chart(fig_fft7)




        st.subheader('Subset 17500-22500')
        n_subset7 = n[17500:22500]
        bpf_eeg_subset7 = hpf_eeg2[17500:22500]

        M = len(bpf_eeg_subset7) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpf_eeg_subset_windowed7 = bpf_eeg_subset7 * hamming_window

        
        fft_result7 = fourier_transform(bpf_eeg_subset_windowed7)
        sampling_rate = 125
        fft_freq7 = calculate_frequency(len(bpf_eeg_subset_windowed7), sampling_rate)
        half_point7 = len(fft_freq7) // 2
        fft_freq_half7 = fft_freq7[:half_point7]
        fft_result_half7 = fft_result7[:half_point7]

        fig9= go.Figure(data=go.Scatter(x=n_subset7, y=bpf_eeg_subset_windowed7, mode='lines'))
        fig9.update_layout(height=400,width=1500,title="Subset 17500-225000 with Hamming Window",xaxis_title="n",yaxis_title="Nilai",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft8 = go.Figure(data=go.Scatter(x=fft_freq_half7, y=np.abs(fft_result_half7),mode="lines"))
        fig_fft8.update_layout( height=400,width=1500,title="FFT of Subset",xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but13= st.button('Cek       ', on_click=on_button13_click)
        if st.session_state.button13_clicked:
            st.plotly_chart(fig9)
            st.plotly_chart(fig_fft8)


        st.subheader('FFT Total')

        average_fft = (fft_result + fft_result1 +  fft_result2 + fft_result3 + fft_result4 + fft_result5 + fft_result6 + fft_result7) / 8
        average_freq = (fft_freq + fft_freq1 + fft_freq2 +  fft_freq3 +  fft_freq4 +  fft_freq5 +  fft_freq6 +  fft_freq7)/ 8
        half_point_total = len(average_freq) // 2
        fft_freq_total = average_freq[:half_point_total]
        fft_result_total = average_fft[:half_point_total]
        
        fig_fft = go.Figure(data=go.Scatter(x=fft_freq_total, y=np.abs(fft_result_total), mode='lines'))
        fig_fft.update_layout(
            title="FFT TOTAL",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True)
        )
        but7 = st.button('Lihat FFT', on_click=on_button7_click)
        if st.session_state.button7_clicked:
            st.plotly_chart(fig_fft)
       
        def calculate_mean_power_frequency(freq, psd):
            freq = np.asarray(freq)
            psd = np.asarray(psd)
            mpf = np.sum(freq * psd) / np.sum(psd)
            return mpf
        
        fft_freq_total = fft_freq_half 
        fft_result_total = np.abs(fft_result_half)
        
        mpf = calculate_mean_power_frequency(fft_freq_total, fft_result_total)
        alpha_range = (8, 13)
        beta_range = (13, 30)
        gamma_range = (30, 40)
        def determine_band(mpf):
            if alpha_range[0] <= mpf < alpha_range[1]:
                return 'Alpha'
            elif beta_range[0] <= mpf < beta_range[1]:
                return 'Beta'
            elif gamma_range[0] <= mpf < gamma_range[1]:
                return 'Gamma'
            else:
                return 'Out of defined ranges'
        band = determine_band(mpf)
        but8=st.button('Hasil', on_click=on_button8_click)
        if st.session_state.button8_clicked:
            st.success(f'Mean frequency Power (MPF): {mpf}')
            st.success(f'Rentang MPF: {band}')

if Pilihan == 'ECG':
    st.header('Masukkan Data')
    uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=None)
        df.columns = ['huruf', 'nilai']
        data_a,data_b, data_c = filter_data(df)
        data_a = np.clip(data_a, None, 350e6)
        jumlahdata = int(np.size(data_a))
        fs = 125
        elapsed_time = data_a.index * (1 / fs)

        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=elapsed_time, y=data_a, mode='lines', name='ECG (a)', line=dict(color='blue')))
        fig.update_layout(height=500, width=1500, title="Plot Data ECG",xaxis_title="Elapsed Time",yaxis_title="Nilai")
        but9 = st.button('Lihat Sinyal', on_click=on_button9_click)
        if st.session_state.button9_clicked:
            st.plotly_chart(fig)


        st.header('DWT Level 3')
        
        def dirac(x):
            if x == 0:
                dirac_delta = 1
            else:
                dirac_delta = 0
            result = dirac_delta
            return result
        
        h = []
        g = []
        n_list = []
        
        for n in range(-2, 2):
            n_list.append(n)
            temp_h = 1/8 * (dirac(n-1) + 3*dirac(n) + 3*dirac(n+1) + dirac(n+2))
            h.append(temp_h)
            temp_g = -2 * (dirac(n) - dirac(n+1))
            g.append(temp_g)
            
        
        Hw = np.zeros(20000)
        Gw = np.zeros(20000)
        i_list = []
        fs =240
        for i in range(0,fs + 1):
            i_list.append(i)
            reG = 0
            imG = 0
            reH = 0
            imH = 0
            for k in range(-2, 2):
                reG = reG + g[k + abs(-2)] * np.cos(k * 2 * np.pi * i / fs)
                imG = imG - g[k + abs(-2)] * np.sin(k * 2 * np.pi * i / fs)
                reH = reH + h[k + abs(-2)] * np.cos(k * 2 * np.pi * i / fs)
                imH = imH - h[k + abs(-2)] * np.sin(k * 2 * np.pi * i / fs)
            temp_Hw = np.sqrt((reH**2) + (imH**2))
            temp_Gw = np.sqrt((reG**2) + (imG**2))
            Hw[i] = temp_Hw
            Gw[i] = temp_Gw
        i_list = i_list[0:round(fs/2)+1]

        Q= np.zeros((9, round (fs/2)+1))
        i_list = []
        for i in range(0, round(fs/2)+1):
            i_list.append(i)
            Q[1][i] = Gw[i]
            Q[2][i] = Gw[2*i]*Hw[i]
            Q[3][i] = Gw[4*i]*Hw[2*i]*Hw[i]
            Q[4][i] = Gw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
            Q[5][i] = Gw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
            Q[6][i] = Gw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
            Q[7][i] = Gw [64*i]*Hw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
            Q[8][i] = Gw[128*i]*Hw[64*i]*Hw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
        
        qj = np.zeros ((6, 10000))
        k_list = []
        j = 1
        a = -(round(2**j) + round(2**(j-1)) - 2)
        b = -(1- round(2**(j-1))) + 1
        
        for k in range (a,b):
            k_list.append(k)
            qj[1][k+abs(a)] = -2 * (dirac(k) - dirac(k+1))

        k_list = []
        j = 2
        a = -(round(2**j) + round(2**(j-1))-2)
        b = -(1 - round (2**(j-1))) + 1
        for k in range (a,b):
            k_list.append(k)
            qj[2][k+abs(a)] = -1/4*(dirac (k-1) + 3*dirac(k) + 2*dirac(k+1) - 2*dirac(k+2) - 3*dirac(k+3)- dirac (k+4))

        k_list = []
        j = 3
        a= -(round(2**j) + round (2**(j-1)) -2)
        b= -(1 - round(2**(j-1))) + 1
        for k in range (a,b):
            k_list.append(k)
            qj[3][k+abs(a)] = -1/32*(dirac(k-3) + 3*dirac(k-2) + 6*dirac(k-1) + 10*dirac(k) + 11*dirac(k+1) + 9*dirac(k+2) + 4*dirac (k+3) - 4*dirac (k+4)- 9*dirac(k+5)- 11*dirac (k+6)- 10*dirac (k+7)- 6*dirac (k+8) -3*dirac (k+9) -dirac(k+10))


        k_list = []
        j = 4
        a=-(round(2**j) + round(2**(j-1))-2)
        b = -(1- round(2**(j-1))) + 1
        for k in range (a,b):
            k_list.append(k)
            qj[4][k+abs(a)] = -1/256*(dirac (k-7) + 3*dirac(k-6) + 6*dirac(k-5) + 10*dirac(k-4) + 15*dirac(k-3) + 21*dirac(k-2) + 28*dirac(k-1) + 36*dirac(k) + 41*dirac(k+1) + 43*dirac(k+2)+ 42*dirac(k+3) + 38*dirac(k+4) + 31*dirac (k+5) + 21*dirac (k+6) + 8*dirac(k+7)- 8*dirac(k+8)- 21*dirac (k+9) -31*dirac (k+10) -38*dirac (k+11)- 42*dirac(k+12) -43*dirac(k+13) -41*dirac (k+14) -36*dirac (k+15)- 28*dirac (k+16)- 21*dirac(k+17) -15*dirac(k+18)- 10*dirac (k+19)- 6*dirac (k+20) -3*dirac (k+21) -dirac(k+22))


        k_list = []
        j = 5
        a=-(round(2**j) + round(2**(j-1))-2)
        b= -(1 - round(2**(j-1))) + 1
        for k in range (a,b):
            k_list.append(k)
            qj[5][k+abs(a)] = -1/(512) * (dirac(k-15) + 3*dirac(k-14) + 6*dirac(k-13) + 10*dirac(k-12) + 15*dirac(k-11) + 21*dirac(k-10) + 28*dirac(k-9) + 36*dirac(k-8) + 45*dirac(k-7) + 55*dirac(k-6) + 66*dirac(k-5) + 78*dirac(k-4) + 91*dirac(k-3) + 105*dirac(k-2) + 120*dirac(k-1) + 136*dirac(k) + 149*dirac(k+1) + 159*dirac(k+2) + 166*dirac(k+3) + 170*dirac(k+4) + 171*dirac(k+5) + 169*dirac(k+6) + 164*dirac(k+7) + 156*dirac(k+8) + 145*dirac(k+9) + 131*dirac(k+10) + 114*dirac(k+11) + 94*dirac(k+12) + 71*dirac(k+13) + 45*dirac(k+14) + 16*dirac(k+15) - 16*dirac(k+16) - 45*dirac(k+17) - 71*dirac(k+18) - 94*dirac(k+19) - 114*dirac(k+20) - 131*dirac(k+21) - 145*dirac(k+22) - 156*dirac(k+23) - 164*dirac(k+24) - 169*dirac(k+25) - 171*dirac(k+26) - 170*dirac(k+27) - 166*dirac(k+28) - 159*dirac(k+29) - 149*dirac(k+30) - 136*dirac(k+31) - 120*dirac(k+32) - 105*dirac(k+33) - 91*dirac(k+34) - 78*dirac(k+35) - 66*dirac(k+36) - 55*dirac(k+37) - 45*dirac(k+38) - 36*dirac(k+39) - 28*dirac(k+40)- 21*dirac(k+41) - 15*dirac(k+42) - 10*dirac(k+43) - 6*dirac(k+44) - 3*dirac(k+45) - dirac(k+46))



        mins = 0
        maks = elapsed_time[-1]
        
        T1 = round(2**(1-1)) - 1
        T2 = round(2**(2-1)) - 1
        T3 = round(2**(3-1)) - 1
        T4 = round(2**(4-1)) - 1
        T5 = round(2**(5-1)) - 1
        print('T1 =', T1)
        print('T2 =', T2)
        print('T3 =', T3)
        print('T4 =', T4)
        print('T5 =', T5)


        ecg = data_a
        w2fb = np.zeros((4,100000))
        N = len(data_a)
        n_list = []
        for n in range (0, N):
            n_list.append(n)
            for j in range(1,6):
                w2fb[1][n+T1] = 0
                w2fb[2][n+T2] = 0
                w2fb[3][n+T3] = 0
                a = -(round(2**j) + round(2**(j-1))-2)
                b = -(1- round(2**(j-1)))
                for k in range(a,b+1):
                    index = n - (k + abs(a))
                    if index >= 0 and index < len(ecg):
                        w2fb[1][n+T1] = w2fb[1][n+T1]+qj[1][(k+abs(a))]*ecg[n-(k+abs(a))];
                        w2fb[2][n+T2] = w2fb[2][n+T2]+qj[2][(k+abs(a))]*ecg[n-(k+abs(a))];
                        w2fb[3][n+T3] = w2fb[3][n+T3]+qj[3][(k+abs(a))]*ecg[n-(k+abs(a))];
                        
        
        figs = []
        for i in range(1, 4):
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=elapsed_time, y=w2fb[i][:len(n_list)], mode='lines', name=f'Orde {i}'))
            fig5.update_layout(title=f'Plot Orde {i}', xaxis_title='elapsed_time',  yaxis_title='Nilai', height=400, width=1500)
        figs.append(fig5)
        
        but10 = st.button('cek', on_click=on_button10_click)
        if st.session_state.button10_clicked:
             for fig5 in figs:
                 st.plotly_chart(fig5)


        st.header('Zero Crossing')
                 
        gradien1 = np.zeros(N)
        gradien2 = np.zeros(N)
        gradien3 = np.zeros(N)
        delay = T3
        for k  in range(delay, N-delay):
            gradien3[k] = w2fb[3][k-delay] - w2fb[3][k+delay]

        fig25= go.Figure()
        fig25.add_trace(go.Scatter(x=data_a.index, y=gradien3, mode='lines', name='Gradien 3', line=dict(color='blue')))
        fig25.update_layout(title='Gradien 3', xaxis_title='Time (s)', yaxis_title='Amplitude (V)', height=400, width=1500)
        but2 = st.button('cek ', on_click=on_button2_click)
        if st.session_state.button2_clicked:
            st.plotly_chart(fig25)
       


        st.header('Deteksi QRS')
        hasil_QRS = np.zeros(len(elapsed_time))
        for i in range(N):
            if (gradien3[i] > 190e6):
                hasil_QRS[i-(T4+1)] = 400e6
            else:
                hasil_QRS[i-(T4+1)] = 0
                
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=elapsed_time, y=hasil_QRS, mode='lines', name='QRS Detection', line=dict(color='blue')))
        fig6.add_trace(go.Scatter(x=elapsed_time, y=data_a, mode='lines', name='Raw ECG', line=dict(color='red')))
        fig6.update_layout(title='QRS Detection', xaxis_title='Time (s)', yaxis_title='Amplitude (V)', height=400, width=1500)
        fig6.update_layout(legend=dict(x=15, y=1, traceorder='normal', font=dict(size=12)))
        but1= st.button('lihat', on_click=on_button1_click)
        if st.session_state.button1_clicked:
            st.plotly_chart(fig6)




        data_a = np.clip(data_a, None, 350e6)
        jumlahdata = int(np.size(data_a))
        fs = 125
        elapsed_time = data_a.index * (1 / fs)
        
        ptp = 0
        waktu = np.zeros(np.size(hasil_QRS))
        selisih = np.zeros(np.size(hasil_QRS))
        for n in range(np.size(hasil_QRS) - 1):
            if hasil_QRS[n] < hasil_QRS[n + 1]:
                waktu[ptp] = n / fs;
                selisih[ptp] = waktu[ptp] - waktu[ptp - 1]
                ptp += 1
        ptp = ptp - 1
        st.success(f'ptp: {ptp}')
        
        
        j = 0
        peak = np.zeros(np.size(hasil_QRS))
        for n in range(np.size(hasil_QRS)-1):
            if hasil_QRS[n] == 400e6 and hasil_QRS[n-1] == 0:
                peak[j] = n
                j += 1
        st.success(f'j: {j}')
        
        
        temp = 0
        interval = np.zeros(np.size(hasil_QRS))
        BPM = np.zeros(np.size(hasil_QRS))
        
        for n in range(ptp):
            interval[n] = (peak[n] - peak[n-1]) / fs
            BPM[n] = 60 / interval[n]
            temp = temp+BPM[n]
            rata = temp / (n - 1)
        st.success(f'rata: {rata}')


        st.header('Time Domain HRV ANALYSIS')
        RR_SDNN=0
        for n in range (ptp):
            RR_SDNN += (((selisih[n])-(60/rata))**2)
        SDNN = math.sqrt (RR_SDNN/ (ptp-1))
        
        
        RR_RMSSD=0
        for n in range (ptp):
            RR_RMSSD += ((selisih[n+1]-selisih[n])**2)
        RMSSD =  math. sqrt (RR_RMSSD/(ptp-1))

            
        NN50 = 0
        for n in range (ptp):
            if (abs(selisih[n+1]-selisih[n])>0.05):
                NN50 +=1  
        pNN50 = (NN50/ (ptp-1)) *100
        
        
        dif = 0
        for n in range (ptp):
            dif += abs(selisih[n]-selisih[n+1])
            RRdif = dif/(ptp-1)
            
        RR_SDSD = 0
        for n in range (ptp):
            RR_SDSD += (((abs(selisih[n]-selisih[n+1]))-RRdif)**2)
        SDSD = math.sqrt(RR_SDSD/(ptp-2))

        data1 = {
            'Metric': ['SDNN', 'RMSSD', 'pNN50', 'SDSD'],
            'Value': [SDNN, RMSSD, pNN50, SDSD]
        }
        df1 = pd.DataFrame(data1)
        st.subheader("HRV Analysis")
        but20= st.button('Lihat       ', on_click=on_button20_click)
        if st.session_state.button20_clicked:
            st.table(df1)

        
            
        bpm_rr = np.zeros(ptp)
        for n in range (ptp):
            bpm_rr[n] = 60/selisih[n]
            if bpm_rr [n]>100:
                bpm_rr[n]=rata
        #bpm_rr
        n = np. arange(0,ptp,1,dtype=int)
        fig7 = go.Figure(data=go.Scatter(x=n, y=bpm_rr, mode='lines'))
        fig7.update_layout(title="TACHOGRAM",xaxis_title="n",yaxis_title="BPM",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        but15= st.button('Tachogram', on_click=on_button15_click)
    
        fig8 = go.Figure()
        fig8.add_trace(go.Histogram(x=bpm_rr, nbinsx=ptp))
        fig8.update_layout(title='Histogram', xaxis_title='BPM', yaxis_title='Frequency')
        fig8.update_layout(xaxis=dict(range=[0, 100]), yaxis=dict(range=[0, 10]))
        if st.session_state.button15_clicked:
            st.plotly_chart(fig7)
            st.plotly_chart(fig8)



        st.header('Frequency Domain HRV ANALYSIS')
        st.subheader('Welchs Method')
        
        bpm_rr_baseline = bpm_rr - 70
        n = np.arange(0, ptp, 1, dtype=int)

        n = np.arange(0, ptp, 1, dtype=int)
        n_subset = n[0:50]
        bpm_rr_baseline_subset = bpm_rr_baseline[0:50]
        M = len(bpm_rr_baseline_subset) - 1
        hamming_window = np.zeros(M+1)
        
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpm_rr_baseline_windowed = bpm_rr_baseline_subset * hamming_window
        
        
        fft_result = fourier_transform(bpm_rr_baseline_windowed)
        sampling_rate = 1
        fft_freq = calculate_frequency(len(bpm_rr_baseline_windowed), sampling_rate)
        half_point = len(fft_freq) // 2
        fft_freq_half = fft_freq[:half_point]
        fft_result_half = fft_result[:half_point]
        
        fig9 = go.Figure(data=go.Scatter(x=n_subset, y=bpm_rr_baseline_windowed, mode='lines'))
        fig9.update_layout(title="TACHOGRAM (Subset 0-49) with Hamming Window",xaxis_title="n",yaxis_title="BPM",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft1 = go.Figure(data=go.Scatter(x=fft_freq_half, y=np.abs(fft_result_half),mode="lines"))
        fig_fft1.update_layout(title="FFT of TACHOGRAM",xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))

        but3 = st.button('Subset 0-49', on_click=on_button3_click)
        if st.session_state.button3_clicked:
            st.plotly_chart(fig9)
            st.plotly_chart(fig_fft1)
        
        
        n = np.arange(0, ptp, 1, dtype=int)
        n_subset1 = n[25:75]
        bpm_rr_baseline_subset1 = bpm_rr_baseline[25:75]
        
        M = len(bpm_rr_baseline_subset1) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpm_rr_baseline_windowed1 = bpm_rr_baseline_subset1 * hamming_window
        
        
        fft_result1 = fourier_transform(bpm_rr_baseline_windowed1)
        sampling_rate = 1
        fft_freq1 = calculate_frequency(len(bpm_rr_baseline_windowed1), sampling_rate)
        half_point1 = len(fft_freq1) // 2
        fft_freq_half1 = fft_freq1[:half_point1]
        fft_result_half1 = fft_result1[:half_point1]
        
        fig10 = go.Figure(data=go.Scatter(x=n_subset1, y=bpm_rr_baseline_windowed1, mode='lines'))
        fig10.update_layout( title="TACHOGRAM (Subset 25-75) with Hamming Window", xaxis_title="n",yaxis_title="BPM",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft2 = go.Figure(data=go.Scatter(x=fft_freq_half1, y=np.abs(fft_result_half1),mode="lines"))
        fig_fft2.update_layout(title="FFT of TACHOGRAM", xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        but4 = st.button('Subset 25-75', on_click=on_button4_click)
        if st.session_state.button4_clicked:
            st.plotly_chart(fig10)
            st.plotly_chart(fig_fft2)
        
        
        n = np.arange(0, ptp, 1, dtype=int)
        n_subset2 = n[50:100] 
        bpm_rr_baseline_subset2 = bpm_rr_baseline[50:100]
        
        M = len(bpm_rr_baseline_subset2) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpm_rr_baseline_windowed2 = bpm_rr_baseline_subset2 * hamming_window
        
        fft_result2 = fourier_transform(bpm_rr_baseline_windowed2)
        sampling_rate = 1
        fft_freq2 = calculate_frequency(len(bpm_rr_baseline_windowed2), sampling_rate)
        half_point2 = len(fft_freq2) // 2
        fft_freq_half2 = fft_freq2[:half_point2]
        fft_result_half2 = fft_result2[:half_point2]
        
        
        fig11 = go.Figure(data=go.Scatter(x=n_subset2, y=bpm_rr_baseline_windowed2, mode='lines'))
        fig11.update_layout( title="TACHOGRAM (Subset 50-100) with Hamming Window", xaxis_title="n", yaxis_title="BPM", xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))

        
        fig_fft3 = go.Figure(data=go.Scatter(x=fft_freq_half2, y=np.abs(fft_result_half2),mode="lines"))
        fig_fft3.update_layout(   title="FFT of TACHOGRAM",  xaxis_title="Frequency (Hz)",  yaxis_title="Magnitude", xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        but5= st.button('Subset 50-100', on_click=on_button5_click)
        if st.session_state.button5_clicked:
            st.plotly_chart(fig11)
            st.plotly_chart(fig_fft3)
        
        
        
        n = np.arange(0, ptp, 1, dtype=int)
        n_subset3 = n[75:125]
        bpm_rr_baseline_subset3 = bpm_rr_baseline[75:125]
        
        M = len(bpm_rr_baseline_subset3) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpm_rr_baseline_windowed3 = bpm_rr_baseline_subset3 * hamming_window
        
        
        fft_result3 = fourier_transform(bpm_rr_baseline_windowed3)
        sampling_rate = 1
        fft_freq3 = calculate_frequency(len(bpm_rr_baseline_windowed3), sampling_rate)
        half_point3 = len(fft_freq3) // 2
        fft_freq_half3 = fft_freq3[:half_point3]
        fft_result_half3 = fft_result3[:half_point3]
        
        fig14= go.Figure(data=go.Scatter(x=n_subset3, y=bpm_rr_baseline_windowed3, mode='lines'))
        fig14.update_layout(title="TACHOGRAM (Subset 75-125) with Hamming Window",xaxis_title="n",yaxis_title="BPM",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
 
        
        fig_fft5 = go.Figure(data=go.Scatter(x=fft_freq_half3, y=np.abs(fft_result_half3),mode="lines"))
        fig_fft5.update_layout( title="FFT of TACHOGRAM", xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        but6= st.button('Subset 75-125', on_click=on_button6_click)
        if st.session_state.button6_clicked:
            st.plotly_chart(fig14)
            st.plotly_chart(fig_fft5)
        
        
        
        n = np.arange(0, ptp, 1, dtype=int)
        n_subset4 = n[100:150]
        bpm_rr_baseline_subset4 = bpm_rr_baseline[100:150]
        
        M = len(bpm_rr_baseline_subset4) - 1
        hamming_window = np.zeros(M+1)
        
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpm_rr_baseline_windowed4 = bpm_rr_baseline_subset4 * hamming_window
        
        
        fft_result4 = fourier_transform(bpm_rr_baseline_windowed4)
        sampling_rate = 1
        fft_freq4 = calculate_frequency(len(bpm_rr_baseline_windowed4), sampling_rate)
        half_point4 = len(fft_freq4) // 2
        fft_freq_half4 = fft_freq4[:half_point4]
        fft_result_half4 = fft_result4[:half_point4]
        
        fig15 = go.Figure(data=go.Scatter(x=n_subset4, y=bpm_rr_baseline_windowed4, mode='lines'))
        fig15.update_layout( title="TACHOGRAM (Subset 100-150) with Hamming Window",xaxis_title="n",yaxis_title="BPM",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))

        
        fig_fft6= go.Figure(data=go.Scatter(x=fft_freq_half4, y=np.abs(fft_result_half4),mode="lines"))
        fig_fft6.update_layout( title="FFT of TACHOGRAM", xaxis_title="Frequency (Hz)",yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
 

        but7= st.button('Subset 100-150', on_click=on_button7_click)
        if st.session_state.button7_clicked:
            st.plotly_chart(fig15)
            st.plotly_chart(fig_fft6)
            
        
        n = np.arange(0, ptp, 1, dtype=int)
        n_subset5 = n[125:175]
        bpm_rr_baseline_subset5 = bpm_rr_baseline[125:175]
        
        M = len(bpm_rr_baseline_subset5) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpm_rr_baseline_windowed5 = bpm_rr_baseline_subset5 * hamming_window
        
        fft_result5 = fourier_transform(bpm_rr_baseline_windowed5)
        sampling_rate = 1
        fft_freq5 = calculate_frequency(len(bpm_rr_baseline_windowed5), sampling_rate)
        half_point5 = len(fft_freq5) // 2
        fft_freq_half5 = fft_freq5[:half_point5]
        fft_result_half5 = fft_result5[:half_point5]
        
        fig16= go.Figure(data=go.Scatter(x=n_subset5, y=bpm_rr_baseline_windowed5, mode='lines'))
        fig16.update_layout(title="TACHOGRAM (Subset 125-175) with Hamming Window",xaxis_title="n",yaxis_title="BPM",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft7 = go.Figure(data=go.Scatter(x=fft_freq_half5, y=np.abs(fft_result_half5),mode="lines"))
        fig_fft7.update_layout(title="FFT of TACHOGRAM", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        but8= st.button('Subset 125-175', on_click=on_button8_click)
        if st.session_state.button8_clicked:
            st.plotly_chart(fig16)
            st.plotly_chart(fig_fft7)
        
        n = np.arange(0, ptp, 1, dtype=int)
        n_subset6 = n[150:200]
        bpm_rr_baseline_subset6 = bpm_rr_baseline[150:200]
        M = len(bpm_rr_baseline_subset6) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpm_rr_baseline_windowed6 = bpm_rr_baseline_subset6 * hamming_window
        
        
        fft_result6 = fourier_transform(bpm_rr_baseline_windowed6)
        sampling_rate = 1
        fft_freq6 = calculate_frequency(len(bpm_rr_baseline_windowed6), sampling_rate)
        half_point6 = len(fft_freq6) // 2
        fft_freq_half6 = fft_freq6[:half_point6]
        fft_result_half6 = fft_result6[:half_point6]
        
        fig17= go.Figure(data=go.Scatter(x=n_subset6, y=bpm_rr_baseline_windowed6, mode='lines'))
        fig17.update_layout(title="TACHOGRAM (Subset 150-200) with Hamming Window",xaxis_title="n",yaxis_title="BPM",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
        
        fig_fft8 = go.Figure(data=go.Scatter(x=fft_freq_half6, y=np.abs(fft_result_half6),mode="lines"))
        fig_fft8.update_layout(title="FFT of TACHOGRAM", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))

        but16= st.button('Subset 150-200', on_click=on_button16_click)
        if st.session_state.button16_clicked:
            st.plotly_chart(fig17)
            st.plotly_chart(fig_fft8)
        
        n = np.arange(0, ptp, 1, dtype=int)
        n_subset7 = n[175:225]
        bpm_rr_baseline_subset7 = bpm_rr_baseline[175:225]
        
        M = len(bpm_rr_baseline_subset7) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpm_rr_baseline_windowed7 = bpm_rr_baseline_subset7 * hamming_window
        
        fft_result7 = fourier_transform(bpm_rr_baseline_windowed7)
        sampling_rate = 1
        fft_freq7 = calculate_frequency(len(bpm_rr_baseline_windowed7), sampling_rate)
        half_point7 = len(fft_freq7) // 2
        fft_freq_half7 = fft_freq7[:half_point7]
        fft_result_half7 = fft_result7[:half_point7]
        
        fig18= go.Figure(data=go.Scatter(x=n_subset7, y=bpm_rr_baseline_windowed7, mode='lines'))
        fig18.update_layout(title="TACHOGRAM (Subset 175-225) with Hamming Window",xaxis_title="n",yaxis_title="BPM",xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))

        
        fig_fft9 = go.Figure(data=go.Scatter(x=fft_freq_half7, y=np.abs(fft_result_half7),mode="lines"))
        fig_fft9.update_layout( title="FFT of TACHOGRAM", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))


        but17= st.button('Subset 175-225', on_click=on_button17_click)
        if st.session_state.button17_clicked:
            st.plotly_chart(fig18)
            st.plotly_chart(fig_fft9)
        
        n = np.arange(0, ptp, 1, dtype=int)
        n_subset8 = n[200:250]
        bpm_rr_baseline_subset8 = bpm_rr_baseline[200:250]
        M = len(bpm_rr_baseline_subset8) - 1
        hamming_window = np.zeros(M+1)
        for i in range(M+1):
            hamming_window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / M)
        bpm_rr_baseline_windowed8 = bpm_rr_baseline_subset8 * hamming_window
        
        
        fft_result8 = fourier_transform(bpm_rr_baseline_windowed8)
        sampling_rate = 1
        fft_freq8 = calculate_frequency(len(bpm_rr_baseline_windowed8), sampling_rate)
        half_point8 = len(fft_freq8) // 2
        fft_freq_half8 = fft_freq8[:half_point8]
        fft_result_half8 = fft_result8[:half_point8]
        
        fig19 = go.Figure(data=go.Scatter(x=n_subset8, y=bpm_rr_baseline_windowed8, mode='lines'))
        fig19.update_layout( title="TACHOGRAM (Subset 200-250) with Hamming Window", xaxis_title="n", yaxis_title="BPM", xaxis=dict(showline=True, showgrid=True),yaxis=dict(showline=True, showgrid=True))
     
        
        fig_fft10= go.Figure(data=go.Scatter(x=fft_freq_half8, y=np.abs(fft_result_half8),mode="lines"))
        fig_fft10.update_layout( title="FFT of TACHOGRAM", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", xaxis=dict(showline=True, showgrid=True), yaxis=dict(showline=True, showgrid=True))
  

        but18= st.button('Subset 200-250', on_click=on_button17_click)
        if st.session_state.button17_clicked:
            st.plotly_chart(fig19)
            st.plotly_chart(fig_fft10)
        
        
        FFT_TOTAL = (fft_result + fft_result1 +  fft_result2 + fft_result3 + fft_result4 + fft_result5 + fft_result6 + fft_result7) / 8
        FFT_FREQ_TOTAL = (fft_freq + fft_freq1 + fft_freq2 +  fft_freq3 +  fft_freq4 +  fft_freq5 +  fft_freq6 +  fft_freq7)/ 8
        
        half_point_total = len(FFT_FREQ_TOTAL) // 2
        fft_freq_total = FFT_FREQ_TOTAL[:half_point_total]
        fft_result_total = FFT_TOTAL[:half_point_total]
        
        fig_fft20 = go.Figure(data=go.Scatter(x=fft_freq_total, y=np.abs(fft_result_total), mode='lines'))
        fig_fft20.update_layout( title="FFT of TACHOGRAM", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", xaxis=dict(showline=True, showgrid=True), yaxis=dict(showline=True, showgrid=True))

        
        
        def manual_interpolation(x, xp, fp):
            return np.interp(x, xp, fp)
            
        x_ulf = np.linspace(0.001, 0.003, 99)
        x_vlf = np.linspace(0.003, 0.04, 99)
        x_lf = np.linspace(0.04, 0.15, 99)
        x_hf = np.linspace(0.15, 0.4, 99)
        
        y_ulf = manual_interpolation(x_ulf, fft_freq_total, np.abs(fft_result_total))
        y_vlf = manual_interpolation(x_vlf, fft_freq_total, np.abs(fft_result_total))
        y_lf = manual_interpolation(x_lf, fft_freq_total, np.abs(fft_result_total))
        y_hf = manual_interpolation(x_hf, fft_freq_total, np.abs(fft_result_total))
        
        
        fig20= go.Figure()
        fig20.add_trace(go.Scatter(x=fft_freq_total,y=np.abs(fft_result_total),mode='lines',line=dict(color='black', width=0.3),name='FFT Spectrum'))
        
        
        fig20.add_trace(go.Scatter( x=x_ulf, y=y_ulf, fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.2)', line=dict(color='rgba(0, 0, 255, 0.5)'), name='ULF'))
        
        fig20.add_trace(go.Scatter( x=x_vlf, y=y_vlf, fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='rgba(255, 0, 0, 0.5)'), name='VLF'))
        
        
        fig20.add_trace(go.Scatter( x=x_lf, y=y_lf, fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='rgba(255, 165, 0, 0.5)'), name='LF'))
        
        fig20.add_trace(go.Scatter(  x=x_hf,  y=y_hf,  fill='tozeroy',  fillcolor='rgba(0, 128, 0, 0.2)',  line=dict(color='rgba(0, 128, 0, 0.5)'),  name='HF'))
        fig20.update_layout( title="FFT Spectrum (Welch's periodogram)", xaxis_title="Frequency (Hz)",yaxis_title="Density",xaxis=dict(range=[0, 0.5]), yaxis=dict(range=[0, max(np.abs(fft_result_total))]), legend=dict(x=0.8, y=0.95))


        but19= st.button('FFT Total', on_click=on_button19_click)
        if st.session_state.button19_clicked:
            st.plotly_chart(fig_fft20)
            st.plotly_chart(fig20)
        
        
        def trapezoidal_rule(y, x):
            return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)
            
        TP = trapezoidal_rule(np.abs(fft_result_total), fft_freq_total)
        ULF = trapezoidal_rule(y_ulf, x_ulf)
        VLF = trapezoidal_rule(y_vlf, x_vlf)
        LF = trapezoidal_rule(y_lf, x_lf)
        HF = trapezoidal_rule(y_hf, x_hf)
        total_power = VLF + LF + HF
        LF_norm = LF / (total_power - VLF)
        HF_norm = HF / (total_power - VLF)
        LF_HF = LF / HF

        data = {
            'Metric': ['Total Power (TP)', 'VLF', 'LF', 'HF', 'Normalized LF', 'Normalized HF', 'LF/HF ratio'],
            'Value': [TP, VLF, LF, HF, LF_norm, HF_norm, LF_HF]
        }
        df = pd.DataFrame(data)
        st.title("HRV Analysis Results")
        but11= st.button('Lihat  ', on_click=on_button11_click)
        if st.session_state.button11_clicked:
            st.table(df)


        def determine_category(LF_norm, HF_norm, LF_HF):
            if LF_norm < 0.2 and HF_norm < 0.2:
                return 1  # Low - Low
            elif LF_norm >= 0.2 and LF_norm <= 0.6 and HF_norm < 0.2:
                return 2  # Normal - Low
            elif LF_norm > 0.6 and HF_norm < 0.2:
                return 3  # High - Low
            elif LF_norm < 0.2 and HF_norm >= 0.2 and HF_norm <= 0.6:
                return 4  # Low - Normal
            elif LF_norm >= 0.2 and LF_norm <= 0.6 and HF_norm >= 0.2 and HF_norm <= 0.6:
                return 5  # Normal - Normal
            elif LF_norm > 0.6 and HF_norm >= 0.2 and HF_norm <= 0.6:
                return 6  # High - Normal
            elif LF_norm < 0.2 and HF_norm > 0.6:
                return 7  # Low - High
            elif LF_norm >= 0.2 and LF_norm <= 0.6 and HF_norm > 0.6:
                return 8  # Normal - High
            elif LF_norm > 0.6 and HF_norm > 0.6:
                return 9  # High - High
            else:
                return 0  # Undefined
        
        

        
        data = [
            [7, 8, 9],
            [4, 5, 6],
            [1, 2, 3]
        ]
        
        plt.figure(figsize=(6, 6))
        
        ax = sns.heatmap(data, annot=True, fmt="d", cmap="magma", cbar=False, linewidths=.5)
        coordinates = {
            1: (2, 0),
            2: (2, 1),
            3: (2, 2),
            4: (1, 0),
            5: (1, 1),
            6: (1, 2),
            7: (0, 0),
            8: (0, 1),
            9: (0, 2)
        }

        st.subheader('Autonomic Balance Diagram')
        category = determine_category(LF_norm, HF_norm, LF_HF)
        coord = coordinates.get(category, None)
        
        if coord:
            ax.plot(coord[1] + 0.5, coord[0] + 0.5, 'ko')
        plt.title("Autonomic Balance Diagram")
        plt.xlabel("Sympathetic Level")
        plt.ylabel("Parasympathetic Level")
        plt.xticks([0.5, 1.5, 2.5], ["Low", "Normal", "High"])
        plt.yticks([0.5, 1.5, 2.5], ["High", "Normal", "Low"])

        but12= st.button('Lihat                                     ', on_click=on_button12_click)
        if st.session_state.button12_clicked:
            st.write("Category:", category)
            st.pyplot(plt)


        st.subheader('POINTCARE PLOT')
        temp = 0
        interval = np.zeros(np.size(hasil_QRS))
        BPM = np.zeros(np.size(hasil_QRS))
        
        for n in range(1, ptp):
            interval[n] = (peak[n] - peak[n-1]) * (1/fs)
            BPM[n] = 60 / interval[n]
            temp += BPM[n]
        rata = temp / (n if n != 0 else 1)

        
        def create_poincare_plot(interval):
            x = interval[:-1]
            y = interval[1:]
            
            plt.figure(figsize=(6, 6))
            plt.scatter(x, y, s=5, c='blue', alpha=0.5)
            plt.title('Poincaré Plot of RR Intervals')
            plt.xlabel('RR(n)')
            plt.ylabel('RR(n+1)')
            plt.grid(True)
            #st.pyplot(plt)
        
        
        create_poincare_plot(interval)
        diff_intervals = np.diff(interval)
        mean_interval = np.mean(interval)
        SD1 = np.std(diff_intervals) / np.sqrt(2)
        SD2 = np.sqrt(2 * np.std(interval)**2 - SD1**2)
        
        but13= st.button('Lihat                                 ', on_click=on_button13_click)
        if st.session_state.button13_clicked:
            st.write("Intervals:", interval)
            st.pyplot(plt)
            st.write(f"SD1: {SD1}")
            st.write(f"SD2: {SD2}")
        
        




        
