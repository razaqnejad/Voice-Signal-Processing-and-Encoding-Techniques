import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import sounddevice as sd

# خواندن فایل صوتی
sampling_rate, data = wav.read('voice1.wav')
print('Sampling rate:', sampling_rate)
print('Data type:', data.dtype)
print('Data shape:', data.shape)

# استخراج ویژگی‌های سیگنال
N, no_channels = data.shape
print('Signal length:', N)
channel0 = data[:, 0]
channel1 = data[:, 1]

# تعریف تابع برای ذخیره فایل صوتی
def save_wav(filename, data, samplerate):
    wav.write(filename, samplerate, data)

# تعریف تابع برای پخش فایل صوتی
def play_audio(data, samplerate):
    sd.play(data, samplerate)
    sd.wait()

# تعریف تابع برای تغییر بلندا
def change_volume(data, factor):
    return np.clip(data * factor, -32768, 32767).astype(np.int16)

# تست تغییر بلندا
def test_change_volume(data, sampling_rate):
    # صدای اصلی
    print('Playing Original Soundtrack')
    # play_audio(data, sampling_rate)

    # دو برابر کردن بلندا
    doubled_volume = change_volume(data, 2)
    save_wav('doubled_volume.wav', doubled_volume, sampling_rate)
    print('Playing Doubled Volume')
    # play_audio(doubled_volume, sampling_rate)
    
    # چهار برابر کردن بلندا
    quadrupled_volume = change_volume(data, 4)
    save_wav('quadrupled_volume.wav', quadrupled_volume, sampling_rate)
    print('Playing Quadrupled Volume')
    # play_audio(quadrupled_volume, sampling_rate)

# تغییر تدریجی بلندا در طول زمان
def gradual_volume_change_over_time(data, sampling_rate, start_factor, end_factor):
    total_samples = data.shape[0]
    factors = np.linspace(start_factor, end_factor, total_samples)
    
    modified_data = np.zeros_like(data)
    for i in range(total_samples):
        modified_data[i] = change_volume(data[i], factors[i])
    
    return modified_data

# فشرده‌سازی A-law
def a_law_compress(data, A=87.6):
    abs_data = np.abs(data) / 32768.0
    compressed_data = np.sign(data) * (1 + np.log(A * abs_data)) / (1 + np.log(A)) * 32768.0
    return np.clip(compressed_data, -32768, 32767).astype(np.int16)

# فشرده‌سازی µ-law
def u_law_compress(data, u=255.0):
    abs_data = np.abs(data) / 32768.0
    compressed_data = np.sign(data) * (np.log(1 + u * abs_data)) / (np.log(1 + u)) * 32768.0
    return np.clip(compressed_data, -32768, 32767).astype(np.int16)

# مدولاسیون دلتا
def delta_modulation(data):
    if len(data.shape) == 2:  # بررسی داده‌های استریو
        delta_signal = np.zeros_like(data)
        for channel in range(data.shape[1]):
            predictor = 0
            for i in range(len(data)):
                if data[i, channel] > predictor:
                    delta_signal[i, channel] = 1000
                    predictor += 1000
                else:
                    delta_signal[i, channel] = -1000
                    predictor -= 1000
    else:  # داده‌های مونو
        delta_signal = np.zeros_like(data)
        predictor = 0
        for i in range(len(data)):
            if data[i] > predictor:
                delta_signal[i] = 1000
                predictor += 1000
            else:
                delta_signal[i] = -1000
                predictor -= 1000

    return delta_signal.astype(np.int16)

# کوانتیزاسیون به تعداد بیت مشخص
def quantize(data, bits):
    max_val = 2**(bits - 1) - 1
    step = 32768 // max_val
    quantized_data = (data // step).astype(np.int16) * step
    return quantized_data

# افزایش سرعت صدا با تغییر نرخ نمونه‌برداری
def increase_speed(data, factor):
    return data[::factor]

# تابع اصلی
def main():
    # مرحله 1: تغییر بلندا و تست
    test_change_volume(data, sampling_rate)
  
    # مرحله 2: تغییر تدریجی بلندا در طول زمان و رسم نمودار
    modified_data = gradual_volume_change_over_time(data, sampling_rate, 1/4, 16)
    save_wav('gradual_volume_change_entire_duration.wav', modified_data, sampling_rate)
    print('Playing Audio with Gradual Volume Change Over Entire Duration')
    # play_audio(modified_data, sampling_rate)

    plt.figure(figsize=(12, 6))
    plt.plot(modified_data)
    plt.title('Audio Signal with Gradual Volume Change Over Entire Duration')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()

    # مرحله 3: فشرده‌سازی A-law و µ-law
    a_law_data = a_law_compress(modified_data)
    u_law_data = u_law_compress(modified_data)
    
    # پلات اول برای سیگنال a-law
    plt.subplot(2, 1, 1)
    plt.plot(a_law_data)
    plt.title('A-Law Compressed Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

    # پلات دوم برای سیگنال u-law
    plt.subplot(2, 1, 2)
    plt.plot(u_law_data)
    plt.title('U-Law Compressed Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

    save_wav('a_law_compressed.wav', a_law_data, sampling_rate)
    print('Playing a-law PCM')
    # play_audio(a_law_data, sampling_rate)

    save_wav('u_law_compressed.wav', u_law_data, sampling_rate)
    print('Playing µ-law PCM')
    # play_audio(u_law_data, sampling_rate)

    # مرحله 4: مدولاسیون دلتا
    delta_data = delta_modulation(modified_data)
    save_wav('delta_modulated.wav', delta_data, sampling_rate)
    print('Playing Delta Modulated Audio')
    # play_audio(delta_data, sampling_rate)

    plt.figure(figsize=(12, 6))
    plt.plot(delta_data[50:150])
    plt.title('Delta Modulated Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()

    # مرحله 5: کوانتیزاسیون
    quantized_2bit = quantize(modified_data, 2)
    quantized_4bit = quantize(modified_data, 4)
    quantized_8bit = quantize(modified_data, 8)

    save_wav('quantized_2bit.wav', quantized_2bit, sampling_rate)
    print('Playing 2 Bit Quantized Audio')
    # play_audio(quantized_2bit, sampling_rate)

    save_wav('quantized_4bit.wav', quantized_4bit, sampling_rate)
    print('Playing 4 Bit Quantized Audio')
    # play_audio(quantized_4bit, sampling_rate)

    save_wav('quantized_8bit.wav', quantized_8bit, sampling_rate)
    print('Playing 8 Bit Quantized Audio')
    # play_audio(quantized_8bit, sampling_rate)

    # مرحله 6: افزایش سرعت صدا
    speeded_2x = increase_speed(data, 2)
    speeded_3x = increase_speed(data, 3)

    save_wav('speeded_2x.wav', speeded_2x, sampling_rate)
    print('Playing 2x Audio')
    # play_audio(speeded_2x, sampling_rate)

    save_wav('speeded_3x.wav', speeded_3x, sampling_rate)
    print('Playing 3x Audio')
    # play_audio(speeded_3x, sampling_rate)

if __name__ == "__main__":
    main()
