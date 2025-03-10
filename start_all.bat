@echo off
echo 正在启动假新闻检测系统...
echo.

rem 启动后端API服务器
echo 正在启动后端API服务器...
start cmd /k "cd api && python app.py"

rem 等待2秒，确保API服务器已启动
timeout /t 2

rem 启动前端服务器
echo 正在启动前端服务器...
cd frontend
start cmd /k "npm run serve"

echo.
echo 所有服务已启动！
echo 前端地址: http://localhost:8080
echo 后端API地址: http://localhost:5000
echo.
echo 注意: 关闭此窗口不会停止服务，请手动关闭服务窗口以停止服务。
echo.

pause 