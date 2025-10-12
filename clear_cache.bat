@echo off
echo ========================================
echo 清理Python缓存文件
echo ========================================
echo.

echo 正在删除 __pycache__ 目录...
if exist __pycache__ (
    rmdir /s /q __pycache__
    echo ✅ 已删除 __pycache__ 目录
) else (
    echo ℹ️ 未找到 __pycache__ 目录
)

echo.
echo 正在删除 .pyc 文件...
del /s /q *.pyc 2>nul
if %errorlevel% equ 0 (
    echo ✅ 已删除所有 .pyc 文件
) else (
    echo ℹ️ 未找到 .pyc 文件
)

echo.
echo 正在删除 .pyo 文件...
del /s /q *.pyo 2>nul
if %errorlevel% equ 0 (
    echo ✅ 已删除所有 .pyo 文件
) else (
    echo ℹ️ 未找到 .pyo 文件
)

echo.
echo ========================================
echo 缓存清理完成！
echo ========================================
echo.
echo 请重启 ComfyUI 以加载最新的代码！
echo.
pause

