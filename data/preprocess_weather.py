# ==============================================================================
# 文件路径: preprocess_weather.py
# 描述: 杭州 2024 气象数据预处理与重采样工具
# 作用: 将 1小时级、日常单位的气象数据，转化为 10分钟级、国际标准单位(SI)的输入格式。
# ==============================================================================

import pandas as pd
import numpy as np
import os

def calculate_absolute_humidity(t_celsius, rh_percent):
    """
    根据温度和相对湿度计算绝对含湿量 (混合比 kg/kg)
    基于 Magnus-Tetens 公式计算饱和水汽压
    """
    # 计算饱和水汽压 (hPa)
    p_sat = 6.112 * np.exp((17.67 * t_celsius) / (t_celsius + 243.5))
    # 实际水汽压 (hPa)
    p_v = p_sat * (rh_percent / 100.0)
    # 标准大气压 (hPa)
    p_atm = 1013.25
    # 绝对湿度 (kg_water / kg_dry_air)
    abs_humidity = 0.622 * p_v / (p_atm - p_v)
    return abs_humidity

def ppm_to_kg_m3(ppm, t_celsius):
    """
    基于理想气体状态方程，将 CO2 的体积浓度 (ppm) 转换为质量浓度 (kg/m^3)
    """
    # CO2 在 0℃, 1个标准大气压下的密度约为 1.977 kg/m^3
    # 根据查理定律进行温度修正
    density_co2 = 1.977 * (273.15 / (273.15 + t_celsius))
    # 转换为质量浓度
    mass_concentration = ppm * 1e-6 * density_co2
    return mass_concentration

def process_weather_data(input_csv, output_csv):
    print(f"🔄 正在读取原始气象数据: {input_csv}")
    df = pd.read_csv(input_csv,encoding='gbk')
    
    # 1. 解析时间并设置为索引
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # 2. 从 1 小时重采样为 10 分钟，并进行线性插值平滑
    print("⏱️ 正在将数据从 1 小时级插值到 10 分钟级...")
    # '10min' 也可以写成 '10T' 视 pandas 版本而定
    df_resampled = df.resample('10min').interpolate(method='linear')
    
    # 3. 科学单位转换与表头重命名
    print("🔬 正在进行热力学单位转换 (ppm -> kg/m3, RH% -> kg/kg)...")
    
    # 提取插值后的临时列
    temp_c = df_resampled['T(℃)']
    co2_ppm = df_resampled['CO2(ppm)']
    rh_pct = df_resampled['RH(%)']
    
    # 创建目标列
    df_final = pd.DataFrame(index=df_resampled.index)
    df_final['Temp_out'] = temp_c
    df_final['CO2_out'] = ppm_to_kg_m3(co2_ppm, temp_c)
    df_final['Hum_out'] = calculate_absolute_humidity(temp_c, rh_pct)
    
    # 4. 确保目标目录存在并保存
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_final.to_csv(output_csv, index=False)
    
    total_steps = len(df_final)
    print(f"✅ 处理完成！已生成 {total_steps} 个 10 分钟级控制步的气象数据。")
    print(f"💾 文件已保存至: {output_csv}")

if __name__ == "__main__":
    # 请确保你的杭州数据放在项目根目录下，或者修改这里的路径
    input_file = "weather_hangzhou_2024.csv" 
    output_file = "data/weather_TMY.csv"
    
    if os.path.exists(input_file):
        process_weather_data(input_file, output_file)
    else:
        print(f"❌ 找不到输入文件 {input_file}，请检查路径。")