# 讯飞星火大模型API集成
import requests
import json
import time
import base64
import logging
import hashlib

class ExplanationGenerator:
    def __init__(self):
        # 讯飞星火API配置
        self.host = "https://spark-api-open.xf-yun.com/v1/chat/completions"
        self.api_key = "JXhEvuqgydysdhNpAudL:yUypnQjpWoFJcTgAvvbw"  # 新的API KEY
        self.service_name = "fakenews"
        self.logger = logging.getLogger("explanation_generator")
        # 添加默认模型参数
        self.model = "4.0Ultra"
        
        # 初始化解释缓存
        self.explanation_cache = {}
        self.max_cache_size = 1000  # 最大缓存条目数
        self.logger.info("解释缓存已初始化，最大缓存数: 1000")
        
    def generate_explanation(self, news_text, prediction):
        """
        使用讯飞星火大模型生成解释
        
        Args:
            news_text: 新闻文本
            prediction: 模型预测结果

        Returns:
            str: 生成的解释文本，如果生成失败则返回默认文本
        """
        try:
            # 只为虚假新闻生成解释
            if prediction['label'] != '虚假新闻':
                return None
                
            # 计算缓存键（使用新闻文本的哈希值）
            cache_key = hashlib.md5(news_text.encode('utf-8')).hexdigest()
            
            # 检查缓存
            if cache_key in self.explanation_cache:
                self.logger.info(f"使用缓存的解释结果: {cache_key[:8]}...")
                return self.explanation_cache[cache_key]
                
            # 创建提示词
            prompt = self._create_prompt(news_text, prediction)
            
            # 调用API
            explanation = self._call_sparkdesk_api(prompt)
            
            # 如果解释为空，返回默认解释
            if not explanation or len(explanation.strip()) == 0:
                explanation = "系统无法生成详细解释，但此新闻具有虚假信息的特征。"
                
            # 存入缓存
            self.explanation_cache[cache_key] = explanation
            self.logger.info(f"解释结果已缓存: {cache_key[:8]}...")
                
            return explanation
            
        except Exception as e:
            self.logger.error(f"生成解释失败: {str(e)}")
            # 返回默认的解释文本
            return "系统无法生成详细解释，但此新闻具有虚假信息的特征。"
    
    def _create_prompt(self, news_text, prediction):
        """创建提示词"""
        # 处理不同类型的confidence值
        if isinstance(prediction.get("confidence"), dict):
            # 如果confidence是字典，获取虚假新闻的置信度
            confidence = prediction["confidence"].get("虚假新闻", 0.5)
        else:
            # 否则直接使用confidence值
            confidence = prediction.get("confidence", 0.5)
            
        # 如果confidence是字典对象，需要特殊处理
        if not isinstance(confidence, (int, float)):
            confidence = 0.7  # 使用默认值
        
        prompt = f"""你是一个专业的假新闻分析专家，请根据以下新闻文本分析为什么它是一条虚假新闻。
            
新闻文本：{news_text}

我们的检测系统判定这是一条虚假新闻，置信度为{confidence*100:.2f}%。

请从以下几个角度分析该新闻为什么可能是虚假的：
1. 内容特征：如夸张用词、情感煽动、标题党等
2. 逻辑漏洞：如内部逻辑矛盾、因果关系不合理等
3. 事实依据：如缺乏来源引用、无具体数据支持等
4. 专业知识：如与已知科学事实或常识不符等

请给出3-5点简明的分析，每点不超过50字，语言要客观专业。"""

        return prompt
            
    def _call_sparkdesk_api(self, prompt):
        """
        调用讯飞星火大模型API
        
        Args:
            prompt: 提示词

        Returns:
            str: 生成的回答文本
        """
        try:
            # 准备请求数据
            request_data = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 1024
            }
            
            # 构建请求头
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 记录请求开始时间（用于性能分析）
            start_time = time.time()
            
            # 发送请求
            response = requests.post(
                self.host,
                headers=headers,
                json=request_data,
                timeout=10  # 设置10秒超时
            )
            
            # 计算响应时间
            response_time = time.time() - start_time
            self.logger.info(f"API响应时间: {response_time:.2f}秒")
            
            # 检查响应状态
            if response.status_code != 200:
                self.logger.error(f"API调用失败: {response.status_code} {response.text}")
                return None
                
            # 解析响应
            result = response.json()
            
            # 检查是否有错误
            if "error" in result:
                self.logger.error(f"API返回错误: {result['error']}")
                return None
                
            # 提取生成的文本
            try:
                # 根据讯飞星火API的响应格式提取文本
                generated_text = result["choices"][0]["message"]["content"]
                return generated_text
            except (KeyError, IndexError) as e:
                self.logger.error(f"API返回格式异常: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"调用API异常: {str(e)}")
            return None 