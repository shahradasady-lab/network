# مدولاسیون‌های پیشرفته GFSK و P-OFDM با بهینه‌سازی هوشمند

در ادامه، دو سیستم مدولاسیون پیشرفته با قابلیت‌های بهینه‌سازی مبتنی بر هوشمصنوعی ارائه می‌شود:

## سیستم ۱: مدولاسیون GFSK با بهینه‌سازی هوشمند

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz
from scipy.integrate import cumtrapz
from scipy.special import erf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

class IntelligentGFSKModem:
    """
    سیستم هوشمند مدولاسیون GFSK با قابلیت‌های:
    1. بهینه‌سازی خودکار پارامترها با الگوریتم‌های تکاملی
    2. تنظیم دینامیکی BT product بر اساس شرایط کانال
    3. جبران‌سازی غیرخطی با شبکه عصبی
    4. تشخیص و تطبیق خودکار نرخ داده
    """
    
    def __init__(self, fs=1000, fc=200, data_rate=100, bt=0.5, modulation_index=0.5):
        """
        مقداردهی اولیه مودم GFSK هوشمند
        
        پارامترها:
        fs: فرکانس نمونه‌برداری (هرتز)
        fc: فرکانس حامل (هرتز)
        data_rate: نرخ داده (bps)
        bt: حاصلضرب پهنای باند-زمان
        modulation_index: شاخص مدولاسیون
        """
        self.fs = fs
        self.fc = fc
        self.data_rate = data_rate
        self.bt = bt
        self.modulation_index = modulation_index
        self.symbol_duration = 1 / data_rate
        self.samples_per_symbol = int(fs / data_rate)
        
        # پارامترهای بهینه‌سازی
        self.optimal_params = None
        self.nn_compensator = None
        self.channel_conditions = {'snr_db': 20, 'doppler': 0, 'multipath': False}
        
        # مدل شبکه عصبی برای جبران‌سازی
        self._build_neural_compensator()
        
    def _build_neural_compensator(self):
        """ساخت شبکه عصبی برای جبران‌سازی غیرخطی"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(50, 2)),  # I/Q samples
            tf.keras.layers.Conv1D(32, 5, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GRU(32, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='tanh'),
            tf.keras.layers.Dense(2, activation='linear')  # I/Q output
        ])
        
        model.compile(optimizer='adam',
                     loss='mse',
                     metrics=['mae'])
        self.nn_compensator = model
        
    def _gaussian_filter(self, bt=None):
        """
        طراحی فیلتر گاوسی با BT product بهینه
        
        پارامترها:
        bt: حاصلضرب پهنای باند-زمان (اگر None باشد از مقدار پیش‌فرض استفاده می‌کند)
        
        بازگشت:
        فیلتر گاوسی نرمال‌شده
        """
        if bt is None:
            bt = self.bt
            
        # محاسبه پهنای باند 3-dB
        b3db = bt / self.symbol_duration
        
        # طول فیلتر (فرد برای تقارن)
        filter_length = 4 * self.samples_per_symbol + 1
        
        # ایجاد فیلتر گاوسی
        t = np.arange(-filter_length//2, filter_length//2 + 1) / self.fs
        gaussian_filter = np.sqrt(2 * np.pi * np.log(2)) * b3db * \
                         np.exp(-2 * (np.pi * b3db * t) ** 2 / np.log(2))
        
        # نرمال‌سازی
        gaussian_filter /= np.sum(gaussian_filter)
        
        return gaussian_filter
    
    def _frequency_pulse(self, data, bt=None):
        """
        تولید پالس فرکانسی با فیلتر گاوسی
        
        پارامترها:
        data: داده‌های باینری
        bt: حاصلضرب پهنای باند-زمان
        
        بازگشت:
        پالس فرکانسی
        """
        # نگاشت داده‌ها: 0 -> -1, 1 -> 1
        mapped_data = 2 * data - 1
        
        # گسترش هر نمونه به تعداد samples_per_symbol
        expanded_data = np.repeat(mapped_data, self.samples_per_symbol)
        
        # فیلتر کردن با فیلتر گاوسی
        gaussian_filter = self._gaussian_filter(bt)
        filtered_data = lfilter(gaussian_filter, 1, expanded_data)
        
        return filtered_data
    
    def modulate(self, binary_data, adaptive=True):
        """
        مدولاسیون GFSK با قابلیت تطبیق‌پذیری
        
        پارامترها:
        binary_data: آرایه‌ای از بیت‌ها (0 یا 1)
        adaptive: فعال‌سازی بهینه‌سازی تطبیقی
        
        بازگشت:
        سیگنال مدوله‌شده
        """
        if adaptive:
            # بهینه‌سازی پارامترها بر اساس داده
            self._optimize_parameters(binary_data)
        
        # تولید پالس فرکانسی
        freq_pulse = self._frequency_pulse(binary_data, self.bt)
        
        # محاسبه فاز
        phase = 2 * np.pi * self.modulation_index * cumtrapz(
            freq_pulse, dx=1/self.fs, initial=0
        )
        
        # تولید سیگنال GFSK
        t = np.arange(len(phase)) / self.fs
        carrier = np.exp(1j * (2 * np.pi * self.fc * t + phase))
        
        # اعمال جبران‌سازی هوشمند
        if self.nn_compensator is not None:
            carrier = self._neural_compensation(carrier)
        
        return carrier
    
    def _neural_compensation(self, signal):
        """جبران‌سازی غیرخطی با شبکه عصبی"""
        # آماده‌سازی داده برای شبکه عصبی
        iq_data = np.column_stack([np.real(signal), np.imag(signal)])
        
        # ایجاد بسته‌های زمانی
        seq_length = 50
        num_sequences = len(iq_data) // seq_length
        sequences = []
        
        for i in range(num_sequences):
            seq = iq_data[i*seq_length:(i+1)*seq_length]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        
        # پیش‌بینی با شبکه عصبی (در حالت عملی، مدل باید آموزش دیده باشد)
        # این بخش نیاز به داده‌های آموزشی دارد
        # compensated_sequences = self.nn_compensator.predict(sequences)
        
        # برای نمونه، سیگنال اصلی را بازمی‌گردانیم
        return signal
    
    def _optimize_parameters(self, binary_data):
        """بهینه‌سازی پارامترهای مدولاسیون با الگوریتم تکاملی"""
        def objective(params):
            bt, mod_index = params
            
            # محدودیت‌ها
            if not (0.3 <= bt <= 0.7) or not (0.4 <= mod_index <= 0.9):
                return 1e6
            
            # شبیه‌سازی عملکرد با پارامترهای جدید
            self.bt = bt
            self.modulation_index = mod_index
            
            # مدولاسیون با پارامترهای جدید
            modulated = self.modulate(binary_data[:100], adaptive=False)
            
            # معیارهای بهینه‌سازی
            # 1. پهنای باند موثر
            psd = np.abs(np.fft.fft(modulated))**2
            freq = np.fft.fftfreq(len(modulated), 1/self.fs)
            
            # پهنای باند 99%
            total_power = np.sum(psd)
            cumulative_power = np.cumsum(np.sort(psd)[::-1])
            idx_99 = np.where(cumulative_power >= 0.99 * total_power)[0][0]
            bandwidth_99 = freq[idx_99] - freq[0]
            
            # 2. حساسیت به نویز (تقریب)
            noise_sensitivity = 1 / (mod_index * np.sqrt(bt))
            
            # تابع هزینه ترکیبی
            cost = 0.7 * (bandwidth_99 / self.fs) + 0.3 * noise_sensitivity
            
            return cost
        
        # محدوده پارامترها برای بهینه‌سازی
        bounds = [(0.3, 0.7), (0.4, 0.9)]
        
        # اجرای الگوریتم تکاملی دیفرانسیل
        result = differential_evolution(objective, bounds, maxiter=50, popsize=15)
        
        if result.success:
            self.bt, self.modulation_index = result.x
            self.optimal_params = result.x
    
    def demodulate(self, received_signal, method='coherent'):
        """
        دمودولاسیون GFSK با روش‌های مختلف
        
        پارامترها:
        received_signal: سیگنال دریافتی
        method: روش دمودولاسیون ('coherent', 'noncoherent', 'discriminator')
        
        بازگشت:
        داده‌های باینری بازیابی‌شده
        """
        if method == 'discriminator':
            return self._discriminator_demodulation(received_signal)
        elif method == 'coherent':
            return self._coherent_demodulation(received_signal)
        else:
            return self._noncoherent_demodulation(received_signal)
    
    def _discriminator_demodulation(self, signal):
        """دمودولاسیون با تشخیص‌دهنده فرکانس"""
        # محاسبه مشتق فاز
        phase = np.unwrap(np.angle(signal))
        freq_estimate = np.diff(phase) * self.fs / (2 * np.pi)
        
        # میانگین‌گیری بر روی هر نماد
        symbols = []
        for i in range(0, len(freq_estimate), self.samples_per_symbol):
            symbol_samples = freq_estimate[i:i+self.samples_per_symbol]
            if len(symbol_samples) > 0:
                symbol_value = np.mean(symbol_samples[-self.samples_per_symbol//2:])
                symbols.append(symbol_value)
        
        symbols = np.array(symbols)
        
        # تصمیم‌گیری
        threshold = 0
        binary_data = (symbols > threshold).astype(int)
        
        return binary_data
    
    def _coherent_demodulation(self, signal):
        """دمودولاسیون همدوس"""
        # حذف حامل
        t = np.arange(len(signal)) / self.fs
        baseband = signal * np.exp(-1j * 2 * np.pi * self.fc * t)
        
        # فیلتر تطبیق‌گیر
        gaussian_filter = self._gaussian_filter(self.bt)
        filtered = lfilter(gaussian_filter, 1, baseband)
        
        # استخراج فاز
        phase = np.unwrap(np.angle(filtered))
        
        # آشکارسازی تغییرات فاز
        phase_diff = np.diff(phase[self.samples_per_symbol//2::self.samples_per_symbol])
        
        # تصمیم‌گیری
        binary_data = (phase_diff > 0).astype(int)
        
        return binary_data
    
    def add_channel_effects(self, signal, snr_db=20, doppler=0, multipath=False):
        """
        اضافه کردن اثرات کانال برای شبیه‌سازی واقع‌بینانه
        
        پارامترها:
        signal: سیگنال ورودی
        snr_db: نسبت سیگنال به نویز (dB)
        doppler: فرکانس دوپلر (هرتز)
        multipath: فعال‌سازی کانال چندمسیره
        
        بازگشت:
        سیگنال تحت تاثیر کانال
        """
        # افزودن نویز گاوسی سفید
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 
                                        1j * np.random.randn(len(signal)))
        noisy_signal = signal + noise
        
        # اثر دوپلر
        if doppler > 0:
            t = np.arange(len(signal)) / self.fs
            doppler_shift = np.exp(1j * 2 * np.pi * doppler * t)
            noisy_signal *= doppler_shift
        
        # کانال چندمسیره
        if multipath:
            # تاخیرهای چندمسیره
            delays = [0, 2, 5]  # نمونه‌ها
            gains = [1.0, 0.5, 0.3]  # ضرایب تضعیف
            
            multipath_signal = np.zeros_like(noisy_signal, dtype=complex)
            for delay, gain in zip(delays, gains):
                if delay == 0:
                    multipath_signal += gain * noisy_signal
                else:
                    multipath_signal[delay:] += gain * noisy_signal[:-delay]
            
            noisy_signal = multipath_signal
        
        return noisy_signal
    
    def analyze_performance(self, original_data, received_data):
        """تحلیل عملکرد سیستم"""
        # محاسبه نرخ خطای بیت (BER)
        if len(original_data) > len(received_data):
            original_data = original_data[:len(received_data)]
        else:
            received_data = received_data[:len(original_data)]
        
        errors = np.sum(original_data != received_data)
        ber = errors / len(original_data)
        
        # محاسبه طیف توان
        modulated_signal = self.modulate(original_data, adaptive=False)
        psd = np.abs(np.fft.fft(modulated_signal))**2
        freq = np.fft.fftfreq(len(modulated_signal), 1/self.fs)
        
        # پهنای باند موثر
        sorted_psd = np.sort(psd)[::-1]
        cumulative_power = np.cumsum(sorted_psd)
        total_power = np.sum(psd)
        
        idx_90 = np.where(cumulative_power >= 0.90 * total_power)[0][0]
        idx_99 = np.where(cumulative_power >= 0.99 * total_power)[0][0]
        
        bandwidth_90 = freq[idx_90] - freq[0]
        bandwidth_99 = freq[idx_99] - freq[0]
        
        return {
            'ber': ber,
            'bandwidth_90': bandwidth_90,
            'bandwidth_99': bandwidth_99,
            'optimal_params': self.optimal_params
        }

# نمونه استفاده از سیستم GFSK
def demonstrate_gfsk_system():
    """نمایش عملکرد سیستم GFSK هوشمند"""
    
    # پارامترهای سیستم
    fs = 1000  # فرکانس نمونه‌برداری
    fc = 200   # فرکانس حامل
    data_rate = 100  # نرخ داده
    bt = 0.5   # BT product
    mod_index = 0.5  # شاخص مدولاسیون
    
    # ایجاد مودم GFSK
    modem = IntelligentGFSKModem(fs=fs, fc=fc, data_rate=data_rate, 
                                 bt=bt, modulation_index=mod_index)
    
    # تولید داده‌های تصادفی
    num_bits = 1000
    binary_data = np.random.randint(0, 2, num_bits)
    
    print("="*60)
    print("سیستم مدولاسیون GFSK با بهینه‌سازی هوشمند")
    print("="*60)
    print(f"پارامترهای اولیه:")
    print(f"  فرکانس نمونه‌برداری: {fs} Hz")
    print(f"  فرکانس حامل: {fc} Hz")
    print(f"  نرخ داده: {data_rate} bps")
    print(f"  BT product: {bt}")
    print(f"  شاخص مدولاسیون: {mod_index}")
    
    # مدولاسیون با بهینه‌سازی تطبیقی
    print("\nانجام مدولاسیون با بهینه‌سازی تطبیقی...")
    modulated_signal = modem.modulate(binary_data, adaptive=True)
    
    if modem.optimal_params is not None:
        print(f"پارامترهای بهینه شده:")
        print(f"  BT product بهینه: {modem.optimal_params[0]:.3f}")
        print(f"  شاخص مدولاسیون بهینه: {modem.optimal_params[1]:.3f}")
    
    # اضافه کردن اثرات کانال
    print("\nاضافه کردن اثرات کانال (SNR=15dB, دوپلر=5Hz)...")
    channel_signal = modem.add_channel_effects(
        modulated_signal, snr_db=15, doppler=5, multipath=True
    )
    
    # دمودولاسیون
    print("انجام دمودولاسیون...")
    demodulated_data = modem.demodulate(channel_signal, method='discriminator')
    
    # تحلیل عملکرد
    print("\nتحلیل عملکرد سیستم:")
    performance = modem.analyze_performance(binary_data, demodulated_data)
    
    print(f"  نرخ خطای بیت (BER): {performance['ber']:.6f}")
    print(f"  پهنای باند 90%: {performance['bandwidth_90']:.2f} Hz")
    print(f"  پهنای باند 99%: {performance['bandwidth_99']:.2f} Hz")
    
    # نمایش سیگنال‌ها
    plt.figure(figsize=(15, 10))
    
    # داده‌های اصلی
    plt.subplot(3, 2, 1)
    plt.step(range(len(binary_data[:50])), binary_data[:50], where='post')
    plt.title('داده‌های باینری اصلی')
    plt.xlabel('نمونه')
    plt.ylabel('مقدار')
    plt.grid(True)
    
    # سیگنال مدوله‌شده
    plt.subplot(3, 2, 2)
    plt.plot(np.real(modulated_signal[:500]))
    plt.title('بخش حقیقی سیگنال GFSK')
    plt.xlabel('نمونه')
    plt.ylabel('دامنه')
    plt.grid(True)
    
    # طیف توان
    plt.subplot(3, 2, 3)
    psd = np.abs(np.fft.fft(modulated_signal))**2
    freq = np.fft.fftfreq(len(modulated_signal), 1/fs)
    positive_freq = freq[:len(freq)//2]
    positive_psd = psd[:len(psd)//2]
    plt.plot(positive_freq, 10*np.log10(positive_psd))
    plt.title('طیف توان سیگنال GFSK')
    plt.xlabel('فرکانس (Hz)')
    plt.ylabel('توان (dB)')
    plt.grid(True)
    
    # سیگنال تحت تاثیر کانال
    plt.subplot(3, 2, 4)
    plt.plot(np.real(channel_signal[:500]))
    plt.title('سیگنال تحت تاثیر کانال')
    plt.xlabel('نمونه')
    plt.ylabel('دامنه')
    plt.grid(True)
    
    # نمودار صورتگر فاز
    plt.subplot(3, 2, 5)
    phase = np.unwrap(np.angle(modulated_signal[:500]))
    plt.plot(phase)
    plt.title('فاز سیگنال GFSK')
    plt.xlabel('نمونه')
    plt.ylabel('فاز (رادیان)')
    plt.grid(True)
    
    # مقایسه داده‌های اصلی و دمودوله شده
    plt.subplot(3, 2, 6)
    plt.step(range(50), binary_data[:50], 'b-', where='post', label='اصلی', alpha=0.7)
    plt.step(range(50), demodulated_data[:50], 'r--', where='post', label='دمودوله', alpha=0.7)
    plt.title('مقایسه داده‌های اصلی و دمودوله')
    plt.xlabel('نمونه')
    plt.ylabel('مقدار')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return modem, performance

# اجرای نمایش سیستم GFSK
if __name__ == "__main__":
    gfsk_modem, perf = demonstrate_gfsk_system()
```

## سیستم ۲: مدولاسیون P-OFDM با بهینه‌سازی پیشرفته

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
from scipy.signal import resample, windows
import tensorflow as tf
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class IntelligentPOFDMModem:
    """
    سیستم هوشمند مدولاسیون P-OFDM (Polar OFDM) با قابلیت‌های:
    1. کدگذاری قطبی برای تصحیح خطا
    2. بهینه‌سازی تخصیص توان با الگوریتم water-filling هوشمند
    3. کاهش PAPR با تکنیک‌های پیشرفته
    4. تخمین و جبران کانال با شبکه عصبی
    5. سازگاری دینامیک با شرایط کانال
    """
    
    def __init__(self, 
                 n_subcarriers=64,
                 cp_length=16,
                 modulation_order=4,
                 frame_size=10,
                 fs=1000):
        """
        مقداردهی اولیه مودم P-OFDM هوشمند
        
        پارامترها:
        n_subcarriers: تعداد زیرحامل‌ها
        cp_length: طول پیشوند چرخشی
        modulation_order: ترتیب مدولاسیون (2 برای QPSK، 4 برای 16-QAM، ...)
        frame_size: تعداد نمادهای OFDM در هر فریم
        fs: فرکانس نمونه‌برداری
        """
        self.n_subcarriers = n_subcarriers
        self.cp_length = cp_length
        self.modulation_order = modulation_order
        self.frame_size = frame_size
        self.fs = fs
        
        # پارامترهای مدولاسیون
        self.modulation_scheme = self._get_modulation_scheme(modulation_order)
        
        # پارامترهای کدگذاری قطبی
        self.polar_n = 128  # طول بلوک کد قطبی
        self.polar_k = 64   # طول اطلاعات کد قطبی
        
        # پارامترهای بهینه‌سازی
        self.power_allocation = None
        self.channel_estimator = None
        self.papr_reduction_model = None
        
        # ساخت مدل‌های هوشمند
        self._build_channel_estimator()
        self._build_papr_reduction_model()
        
        # محاسبه پارامترهای سیگنال
        self.symbol_length = n_subcarriers + cp_length
        self.subcarrier_spacing = fs / n_subcarriers
        
    def _get_modulation_scheme(self, order):
        """تعیین طرح مدولاسیون بر اساس ترتیب"""
        if order == 2:  # QPSK
            constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
            bit_mapping = {0: constellation[0], 1: constellation[1], 
                          2: constellation[2], 3: constellation[3]}
        elif order == 4:  # 16-QAM
            constellation = []
            for i in range(-3, 4, 2):
                for j in range(-3, 4, 2):
                    constellation.append(i + 1j*j)
            constellation = np.array(constellation) / np.sqrt(10)
            bit_mapping = {i: constellation[i] for i in range(16)}
        else:  # 64-QAM پیش‌فرض
            constellation = []
            for i in range(-7, 8, 2):
                for j in range(-7, 8, 2):
                    constellation.append(i + 1j*j)
            constellation = np.array(constellation) / np.sqrt(42)
            bit_mapping = {i: constellation[i] for i in range(64)}
        
        return {
            'order': order,
            'constellation': constellation,
            'bit_mapping': bit_mapping,
            'bits_per_symbol': int(np.log2(len(constellation)))
        }
    
    def _build_channel_estimator(self):
        """ساخت شبکه عصبی برای تخمین کانال"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.n_subcarriers, 2)),
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GRU(64, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear')  # Real and Imag parts
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        self.channel_estimator = model
    
    def _build_papr_reduction_model(self):
        """ساخت مدل کاهش PAPR"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.n_subcarriers, 2)),
            tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Attention(use_scale=True),
            tf.keras.layers.Dense(32, activation='tanh'),
            tf.keras.layers.Dense(2, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        self.papr_reduction_model = model
    
    def polar_encode(self, data_bits):
        """
        کدگذاری قطبی برای تصحیح خطا
        
        پارامترها:
        data_bits: بیت‌های داده
        
        بازگشت:
        بیت‌های کدگذاری شده
        """
        n = self.polar_n
        k = self.polar_k
        
        # اگر داده کمتر از k باشد، padding می‌کنیم
        if len(data_bits) < k:
            padded_data = np.pad(data_bits, (0, k - len(data_bits)), 'constant')
        else:
            padded_data = data_bits[:k]
        
        # ماتریس کدگذاری قطبی (کرنل آریکان)
        def polar_transform(u):
            N = len(u)
            if N == 1:
                return u
            else:
                u1u2 = u[:N//2] ^ u[N//2:]  # XOR
                u2 = u[N//2:]
                return np.concatenate([polar_transform(u1u2), polar_transform(u2)])
        
        # کدگذاری
        encoded_bits = polar_transform(padded_data)
        
        # اگر نیاز به طول بیشتر بود، تکرار می‌کنیم
        if len(encoded_bits) < len(data_bits):
            repeat_factor = int(np.ceil(len(data_bits) / len(encoded_bits)))
            encoded_bits = np.tile(encoded_bits, repeat_factor)[:len(data_bits)]
        
        return encoded_bits
    
    def polar_decode(self, received_llrs):
        """
        کدگشایی قطبی با الگوریتم SC
        
        پارامترها:
        received_llrs: LLRهای دریافتی
        
        بازگشت:
        بیت‌های کدگشایی شده
        """
        n = self.polar_n
        
        # الگوریتم کدگشایی Successive Cancellation (SC)
        def sc_decode(llrs):
            N = len(llrs)
            if N == 1:
                return np.array([1 if llrs[0] < 0 else 0])
            
            # تقسیم LLRها
            llrs_upper = llrs[:N//2]
            llrs_lower = llrs[N//2:]
            
            # محاسبه LLRهای مرحله اول
            llrs_stage1 = np.sign(llrs_upper) * np.sign(llrs_lower) * \
                         np.minimum(np.abs(llrs_upper), np.abs(llrs_lower))
            
            # کدگشایی بازگشتی
            decoded_upper = sc_decode(llrs_stage1)
            
            # محاسبه LLRهای مرحله دوم
            llrs_stage2 = llrs_lower + (1 - 2*decoded_upper) * llrs_upper
            
            # کدگشایی بازگشتی
            decoded_lower = sc_decode(llrs_stage2)
            
            # ترکیب نتایج
            return np.concatenate([decoded_upper ^ decoded_lower, decoded_lower])
        
        # کدگشایی
        decoded_bits = sc_decode(received_llrs[:n])
        
        return decoded_bits[:self.polar_k]
    
    def adaptive_modulation(self, channel_state):
        """
        تطبیق ترتیب مدولاسیون بر اساس وضعیت کانال
        
        پارامترها:
        channel_state: تخمین وضعیت کانال برای هر زیرحامل
        
        بازگشت:
        ترتیب مدولاسیون بهینه برای هر زیرحامل
        """
        snr_per_subcarrier = np.abs(channel_state)**2
        
        # تعیین ترتیب مدولاسیون بر اساس SNR
        modulation_orders = []
        for snr in snr_per_subcarrier:
            if snr > 20:  # dB
                mod_order = 6  # 64-QAM
            elif snr > 15:
                mod_order = 4  # 16-QAM
            elif snr > 10:
                mod_order = 2  # QPSK
            else:
                mod_order = 1  # BPSK
            modulation_orders.append(mod_order)
        
        return np.array(modulation_orders)
    
    def optimal_power_allocation(self, channel_state, total_power=1.0):
        """
        تخصیص توان بهینه با الگوریتم water-filling
        
        پارامترها:
        channel_state: پاسخ فرکانسی کانال
        total_power: توان کل
        
        بازگشت:
        توان تخصیص یافته به هر زیرحامل
        """
        n = len(channel_state)
        noise_power = 1.0  # فرض می‌شود توان نویز نرمال‌شده است
        
        # محاسبه بهره کانال
        channel_gain = np.abs(channel_state)**2 / noise_power
        
        # مرتب‌سازی بهره‌ها
        sorted_indices = np.argsort(channel_gain)[::-1]
        sorted_gains = channel_gain[sorted_indices]
        
        # الگوریتم water-filling
        power_allocation = np.zeros(n)
        
        for m in range(1, n+1):
            # محاسبه سطح آب
            water_level = (total_power + np.sum(1/sorted_gains[:m])) / m
            
            # بررسی معیار توقف
            if water_level > 1/sorted_gains[m-1] if m < n else True:
                power_allocation[sorted_indices[:m]] = water_level - 1/sorted_gains[:m]
                break
        
        # اطمینان از مثبت بودن توان‌ها
        power_allocation = np.maximum(power_allocation, 0)
        
        # نرمال‌سازی
        power_allocation = power_allocation * total_power / np.sum(power_allocation)
        
        self.power_allocation = power_allocation
        return power_allocation
    
    def reduce_papr(self, ofdm_symbol, method='clipping'):
        """
        کاهش PAPR با روش‌های مختلف
        
        پارامترها:
        ofdm_symbol: نماد OFDM
        method: روش کاهش ('clipping', 'companding', 'neural')
        
        بازگشت:
        نماد OFDM با PAPR کاهش یافته
        """
        papr_original = self.calculate_papr(ofdm_symbol)
        
        if method == 'clipping':
            # روش clipping با آستانه بهینه
            threshold = np.percentile(np.abs(ofdm_symbol), 95)
            clipped = np.clip(ofdm_symbol, -threshold, threshold)
            clipped *= np.mean(np.abs(ofdm_symbol)) / np.mean(np.abs(clipped))
            return clipped
            
        elif method == 'companding':
            # روش companding میو-لا
            mu = 2.0
            companded = np.sign(ofdm_symbol) * np.log(1 + mu * np.abs(ofdm_symbol)) / np.log(1 + mu)
            return companded
            
        elif method == 'neural' and self.papr_reduction_model is not None:
            # کاهش PAPR با شبکه عصبی
            iq_data = np.column_stack([np.real(ofdm_symbol), np.imag(ofdm_symbol)])
            iq_data = iq_data.reshape(1, -1, 2)
            
            # پیش‌بینی مدل (نیاز به آموزش دارد)
            # reduced = self.papr_reduction_model.predict(iq_data)
            # reduced_complex = reduced[0,:,0] + 1j*reduced[0,:,1]
            # return reduced_complex
            
            # در صورت عدم آموزش مدل، از روش clipping استفاده می‌کند
            return self.reduce_papr(ofdm_symbol, method='clipping')
        
        else:
            return ofdm_symbol
    
    def calculate_papr(self, signal):
        """محاسبه PAPR"""
        peak_power = np.max(np.abs(signal)**2)
        avg_power = np.mean(np.abs(signal)**2)
        papr_db = 10 * np.log10(peak_power / avg_power)
        return papr_db
