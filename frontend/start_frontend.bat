@echo off
echo 正在启动假新闻检测系统前端...
echo.
echo 请确保已安装Node.js和npm
echo.

cd /d %~dp0
echo 安装依赖...
call npm install

echo.
echo 启动开发服务器...
call npm run serve

pause 