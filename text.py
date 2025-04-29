import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 打开ROOT文件
file = uproot.open("Bs2JpsiPhiData.root")
tree = file["DecayTree"]

# 读取B_M数据
data = tree.arrays(["B_M"], library="pd")
mass = data["B_M"].values

# 定义拟合函数：高斯信号 + 指数背景
def fit_func(x, amp, mean, sigma, a, b):
    gauss = amp * np.exp(-0.5 * ((x - mean) / sigma)**2)
    bkg = np.exp(a + b * x)
    return gauss + bkg

# 创建直方图数据
hist, bin_edges = np.histogram(mass, bins=100, range=(5200, 5550))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 初始参数猜测 (振幅, 均值, 宽度, 背景参数a, 背景参数b)
p0 = [1000, 5366.9, 20, 10, -0.002]


plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 执行拟合
popt, pcov = curve_fit(fit_func, bin_centers, hist, p0=p0)

# 提取拟合参数
amp, mean, sigma, a, b = popt
amp_err, mean_err, sigma_err, a_err, b_err = np.sqrt(np.diag(pcov))

# 绘制结果
plt.figure(figsize=(10, 7))

# 绘制直方图
plt.hist(mass, bins=100, range=(5200, 5550), histtype='step', linewidth=2, label='Data')

# 绘制拟合曲线
x_fit = np.linspace(5200, 5550, 1000)
y_fit = fit_func(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Total fit')

# 单独绘制高斯信号部分
y_gauss = popt[0] * np.exp(-0.5 * ((x_fit - popt[1]) / popt[2])**2)
plt.plot(x_fit, y_gauss, 'g--', linewidth=2, label='Signal')

# 单独绘制背景部分
y_bkg = np.exp(popt[3] + popt[4] * x_fit)
plt.plot(x_fit, y_bkg, 'm--', linewidth=2, label='Background')

# 添加拟合参数文本
fit_info = f"Mean = {mean:.2f} ± {mean_err:.2f} MeV/c²\n"
fit_info += f"Width (σ) = {sigma:.2f} ± {sigma_err:.2f} MeV/c²\n"
plt.text(5400, np.max(hist)*0.8, fit_info, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel('B_s Mass (MeV/c²)', fontsize=12)
plt.ylabel('Event Count', fontsize=12)
plt.title('B_s → J/psi phi Mass Distribution Fit', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Bs_mass_fit.png')

# 打印拟合结果
print("Fit Results:")
print(f"B_s Mass: {mean:.2f} ± {mean_err:.2f} MeV/c²")
print(f"Signal Width: {sigma:.2f} ± {sigma_err:.2f} MeV/c²")
print(f"Signal Amplitude: {amp:.2f} ± {amp_err:.2f}")
print(f"Background Parameter a: {a:.4f} ± {a_err:.4f}")
print(f"Background Parameter b: {b:.6f} ± {b_err:.6f}")

# 计算信噪比 (S/√(S+B))
signal_range = (mean - 2*sigma, mean + 2*sigma)
signal_mask = (mass >= signal_range[0]) & (mass <= signal_range[1])
signal_counts = np.sum(signal_mask)

# 估计信号区域内的背景
x_signal = np.linspace(signal_range[0], signal_range[1], 1000)
background_func = np.exp(a + b * x_signal)
background_estimate = np.trapz(background_func, x_signal) * (signal_range[1]-signal_range[0])/1000

# 估计纯信号
signal_estimate = signal_counts - background_estimate

print(f"\nSignal Region (μ±2σ): [{signal_range[0]:.2f}, {signal_range[1]:.2f}] MeV/c²")
print(f"Estimated Signal Events: {signal_estimate:.0f}")
print(f"Estimated Background Events: {background_estimate:.0f}")
print(f"Signal-to-Noise Ratio (S/√(S+B)): {signal_estimate/np.sqrt(signal_estimate+background_estimate):.2f}")

# 输出最终结果
print("\n======== FINAL RESULT ========")
print(f"The fitted B_s mass is: {mean:.2f} ± {mean_err:.2f} MeV/c²")
print("=============================")