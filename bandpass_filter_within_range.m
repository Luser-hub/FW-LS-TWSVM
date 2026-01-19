function altered_signal = bandpass_filter_within_range(signal, low_freq, high_freq, Fs)
    % 设计带通滤波器
    order = 4;  % 滤波器阶数
    [b, a] = butter(order, [low_freq, high_freq]/(0.5*Fs), 'bandpass');

    % 应用滤波器
    altered_signal = filter(b, a, signal);
end
