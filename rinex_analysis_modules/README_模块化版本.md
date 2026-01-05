# GNSS数据分析器 - 模块化版本

## 概述

本项目将原始的单体 `Rinex_analysis.py` 脚本（8501行）完全模块化拆分，形成了一个结构清晰、便于维护的GNSS数据分析工具。同时保持与原始脚本完全相同的功能和GUI界面，确保用户体验的一致性。

## 项目结构

```
rinex_analysis_modules/
├── __init__.py                    # 包初始化文件和统一导出接口
├── core/                          # 核心分析模块
│   ├── __init__.py
│   ├── config.py                  # GNSS配置参数和系统常数
│   └── analyzer.py                # 核心分析算法（码相差、相位停滞、ISB等）
├── io/                            # 文件输入输出模块  
│   ├── __init__.py
│   ├── rinex_reader.py            # RINEX文件读取器
│   └── rinex_writer.py            # RINEX文件写入器和修正文件生成
├── visualization/                 # 数据可视化模块
│   ├── __init__.py
│   └── plotter.py                 # 图表绘制器（所有图表类型）
└── gui/                           # 图形用户界面模块
    ├── __init__.py
    ├── main_gui.py                # 主界面（文件选择和参数配置）
    ├── preprocessing_gui.py       # 数据预处理界面
    ├── chart_gui.py               # 图表显示界面
    └── report_gui.py              # 报告生成界面

Rinex_Analysis_Modular.py          # 模块化版本主启动脚本
```

## 主要功能

### 1. 核心分析功能 (core模块)
- **配置管理**: GNSS系统频率、波长、卫星信息、观测类型映射等配置参数
- **观测值导数计算**: 计算伪距、相位、多普勒的一阶差分和时间导数
- **相位停滞检测**: 检测载波相位停滞现象，分析停滞期统计信息
- **码相不一致性建模**: CCI建模分析，包括GDOP建模和固定值模型
- **系统间偏差分析**: ISB分析，支持GPS/BDS/Galileo/GLONASS多系统
- **历元间双差**: 计算历元间双差以剔除异常观测值
- **数据清洗**: 自动清洗异常观测数据，支持多种异常检测算法

### 2. 文件处理功能 (io模块)
- **RINEX文件读取**: 支持手机和接收机RINEX观测文件读取
- **数据解析**: 解析观测数据、头部信息、历元时间等
- **修正文件生成**: 生成修正后的RINEX文件
- **处理日志**: 保存分析处理过程的详细日志

### 3. 数据可视化功能 (visualization模块)
- **码相差图表**: 时间序列图、分布直方图、统计对比图、散点图等
- **相位停滞图表**: 停滞期统计图、持续时间分布图、停滞卫星分布等
- **ISB分析图表**: 系统间偏差时间序列、统计分布、多系统对比等
- **CCI建模图表**: GDOP模型拟合图、固定值模型对比图、残差分析图
- **卫星天空图**: 卫星位置分布图、信号强度分布图
- **批量图表生成**: 支持一键生成所有类型的分析图表

### 4. 图形界面功能 (gui模块)
- **三窗口设计**: 主窗口→预处理窗口→图表窗口→报告窗口的流程化界面
- **完整功能界面**: 与原始脚本GUI界面100%一致的用户体验
- **文件选择管理**: 支持手机和接收机RINEX文件选择，智能文件类型检测
- **参数配置**: 完整的分析参数配置选项（CCI模型、阈值设置等）
- **实时进度显示**: 详细的分析进度和状态信息反馈
- **集成结果展示**: 内置图表查看器，支持图表缩放、保存等操作
- **报告生成**: 自动生成分析报告，支持多种输出格式

## 使用方法

### 1. 启动程序
```bash
# 启动模块化版本
python Rinex_Analysis_Modular.py

# 或者直接运行模块
python -m rinex_analysis_modules
```

### 2. 操作流程
1. **选择文件**: 使用"文件"菜单或界面按钮选择RINEX观测文件
   - 手机RINEX文件（必需）
   - 接收机RINEX文件（ISB分析需要）

2. **数据预处理**: 在预处理窗口中配置分析参数
   - CCI建模方式选择（GDOP模型/固定值模型）
   - 异常检测阈值设置
   - 数据清洗选项配置

3. **执行分析**: 选择需要的分析功能
   - 码相不一致性（CCI）建模分析
   - 相位停滞检测分析
   - 系统间偏差（ISB）分析
   - 综合数据质量分析

4. **查看结果**: 在图表窗口查看分析结果
   - 交互式图表查看器
   - 支持图表缩放、平移、保存
   - 批量生成所有图表

5. **生成报告**: 在报告窗口生成完整的分析报告

### 3. 编程接口使用

```python
from rinex_analysis_modules import GNSSAnalyzer, RinexReader, GNSSPlotter

# 初始化组件
reader = RinexReader()
analyzer = GNSSAnalyzer()
plotter = GNSSPlotter()

# 读取RINEX文件
mobile_data = reader.read_rinex_obs("mobile_rinex_file.rnx")
receiver_data = reader.read_rinex_obs("receiver_rinex_file.rnx")  # 可选

# 进行CCI建模分析
cci_results = analyzer.perform_cci_modeling(
    mobile_data, 
    model_type='gdop',  # 或 'fixed'
    threshold=5.0
)

# 相位停滞检测
stagnation_results = analyzer.detect_phase_stagnation(
    mobile_data,
    threshold=0.01
)

# ISB分析（需要接收机数据）
if receiver_data:
    isb_results = analyzer.analyze_isb(mobile_data, receiver_data)

# 生成完整图表
output_dir = "analysis_results"
plotter.plot_cci_analysis(cci_results, output_dir)
plotter.plot_phase_stagnation(stagnation_results, output_dir)
if receiver_data:
    plotter.plot_isb_analysis(isb_results, output_dir)

# 批量生成所有图表
plotter.generate_all_charts(
    mobile_data, 
    cci_results, 
    stagnation_results, 
    isb_results if receiver_data else None,
    output_dir
)
```

## 模块化优势

### 1. 代码结构清晰
- 功能模块化，职责分离
- 便于理解和维护
- 支持独立测试

### 2. 易于扩展
- 新功能可独立添加到相应模块
- 接口统一，便于集成
- 支持插件式扩展

### 3. 重用性强
- 各模块可独立使用
- 便于在其他项目中复用
- 支持不同的调用方式

### 4. 维护友好
- 问题定位更精确
- 修改影响范围可控
- 便于版本管理

## 依赖库

- numpy: 数值计算
- pandas: 数据处理
- matplotlib: 图表绘制
- tkinter: GUI界面（Python标准库）

## 注意事项

1. **字体配置**: 程序会自动检测并配置中文字体（Microsoft YaHei），如显示异常请检查系统字体
2. **文件格式**: 支持标准RINEX 2.x和3.x格式的观测文件，自动识别手机和接收机RINEX
3. **内存使用**: 处理大文件时注意内存占用，建议文件大小不超过500MB
4. **结果保存**: 分析结果和图表会保存到指定的输出目录（默认为`results`文件夹）
5. **依赖环境**: 建议在Python虚拟环境中运行，确保依赖库版本兼容性
6. **GUI兼容性**: 与原始Rinex_analysis.py脚本的GUI界面完全一致，用户无需学习新界面

## 开发信息

- **版本**: 2.0.0
- **作者**: GitHub Copilot  
- **基于**: 原始 Rinex_analysis.py 脚本（8501行）完全模块化重构
- **开发时间**: 2026年1月
- **特点**: 功能完全保持、GUI界面一致、代码结构化、便于维护

## 模块化成果

- ✅ **完全模块化**: 将8501行单体脚本拆分为清晰的模块结构
- ✅ **功能完整性**: 保持原有所有分析功能，无功能缺失
- ✅ **界面一致性**: GUI界面与原始脚本100%一致
- ✅ **代码质量**: 提高代码可读性、可维护性和可扩展性
- ✅ **文档完善**: 提供详细的模块说明和使用文档

## 未来计划

- [ ] 性能优化（大文件处理速度提升）
- [ ] 支持批量文件处理
- [ ] 添加配置文件支持（JSON/YAML格式）
- [ ] 增加单元测试覆盖
- [ ] 支持更多GNSS系统（如北斗三号新信号）
- [ ] 添加命令行接口（CLI）支持