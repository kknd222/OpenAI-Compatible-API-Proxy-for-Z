package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
)

// 配置变量（从环境变量读取）
var (
	UPSTREAM_URL   string
	DEFAULT_KEY    string
	UPSTREAM_TOKEN string
	MODEL_MAP      map[string]string
	PORT           string
	DEBUG_MODE     bool
	DEFAULT_STREAM bool
)

// 思考内容处理策略
const (
	THINK_TAGS_MODE = "strip" // strip: 去除<details>标签；think: 转为<think>标签；raw: 保留原样
)

// 伪装前端头部（来自抓包）
const (
	X_FE_VERSION   = "prod-fe-1.0.70"
	BROWSER_UA     = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0"
	SEC_CH_UA      = "\"Not;A=Brand\";v=\"99\", \"Microsoft Edge\";v=\"139\", \"Chromium\";v=\"139\""
	SEC_CH_UA_MOB  = "?0"
	SEC_CH_UA_PLAT = "\"Windows\""
	ORIGIN_BASE    = "https://chat.z.ai"
)

// 匿名token开关
const ANON_TOKEN_ENABLED = true

// 从环境变量初始化配置
func initConfig() {
	UPSTREAM_URL = getEnv("UPSTREAM_URL", "https://chat.z.ai/api/chat/completions")
	DEFAULT_KEY = getEnv("DEFAULT_KEY", "sk-your-key")
	UPSTREAM_TOKEN = getEnv("UPSTREAM_TOKEN", "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjMxNmJjYjQ4LWZmMmYtNGExNS04NTNkLWYyYTI5YjY3ZmYwZiIsImVtYWlsIjoiR3Vlc3QtMTc1NTg0ODU4ODc4OEBndWVzdC5jb20ifQ.PktllDySS3trlyuFpTeIZf-7hl8Qu1qYF3BxjgIul0BrNux2nX9hVzIjthLXKMWAf9V0qM8Vm_iyDqkjPGsaiQ")
	PORT = getEnv("PORT", "8080")

	modelMapStr := getEnv("MODEL_MAP", "GLM-4.5:0727-360B-API,GLM-4.5V:glm-4.5v")
	MODEL_MAP = make(map[string]string)
	pairs := strings.Split(modelMapStr, ",")
	for _, pair := range pairs {
		kv := strings.SplitN(pair, ":", 2)
		if len(kv) == 2 {
			key := strings.TrimSpace(kv[0])
			value := strings.TrimSpace(kv[1])
			if key != "" && value != "" {
				MODEL_MAP[key] = value
			}
		}
	}

	// 处理PORT格式，确保有冒号前缀
	if !strings.HasPrefix(PORT, ":") {
		PORT = ":" + PORT
	}

	DEBUG_MODE = getEnv("DEBUG_MODE", "true") == "true"
	DEFAULT_STREAM = getEnv("DEFAULT_STREAM", "true") == "true"
}

// 获取环境变量，如果不存在则返回默认值
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// OpenAI 请求结构
type OpenAIRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// 上游请求结构
type UpstreamRequest struct {
	Stream          bool                   `json:"stream"`
	Model           string                 `json:"model"`
	Messages        []Message              `json:"messages"`
	Params          map[string]interface{} `json:"params"`
	Features        map[string]interface{} `json:"features"`
	BackgroundTasks map[string]bool        `json:"background_tasks,omitempty"`
	ChatID          string                 `json:"chat_id,omitempty"`
	ID              string                 `json:"id,omitempty"`
	MCPServers      []string               `json:"mcp_servers,omitempty"`
	ModelItem       struct {
		ID      string `json:"id"`
		Name    string `json:"name"`
		OwnedBy string `json:"owned_by"`
	} `json:"model_item,omitempty"`
	ToolServers []string          `json:"tool_servers,omitempty"`
	Variables   map[string]string `json:"variables,omitempty"`
}

// OpenAI 响应结构
type OpenAIResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage,omitempty"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message,omitempty"`
	Delta        Delta   `json:"delta,omitempty"`
	FinishReason string  `json:"finish_reason,omitempty"`
}

type Delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// 上游SSE响应结构
type UpstreamData struct {
	Type string `json:"type"`
	Data struct {
		DeltaContent string         `json:"delta_content"`
		Phase        string         `json:"phase"`
		Done         bool           `json:"done"`
		Usage        Usage          `json:"usage,omitempty"`
		Error        *UpstreamError `json:"error,omitempty"`
		Inner        *struct {
			Error *UpstreamError `json:"error,omitempty"`
		} `json:"data,omitempty"`
	} `json:"data"`
	Error *UpstreamError `json:"error,omitempty"`
}

type UpstreamError struct {
	Detail string `json:"detail"`
	Code   int    `json:"code"`
}

// 模型列表响应
type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// debug日志函数
func debugLog(format string, args ...interface{}) {
	if DEBUG_MODE {
		log.Printf("[DEBUG] "+format, args...)
	}
}

// 获取匿名token（每次对话使用不同token，避免共享记忆）
func getAnonymousToken() (string, error) {
	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest("GET", ORIGIN_BASE+"/api/v1/auths/", nil)
	if err != nil {
		return "", err
	}
	// 伪装浏览器头
	req.Header.Set("User-Agent", BROWSER_UA)
	req.Header.Set("Accept", "*/*")
	req.Header.Set("Accept-Language", "zh-CN,zh;q=0.9")
	req.Header.Set("X-FE-Version", X_FE_VERSION)
	req.Header.Set("sec-ch-ua", SEC_CH_UA)
	req.Header.Set("sec-ch-ua-mobile", SEC_CH_UA_MOB)
	req.Header.Set("sec-ch-ua-platform", SEC_CH_UA_PLAT)
	req.Header.Set("Origin", ORIGIN_BASE)
	req.Header.Set("Referer", ORIGIN_BASE+"/")

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("anon token status=%d", resp.StatusCode)
	}
	var body struct {
		Token string `json:"token"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return "", err
	}
	if body.Token == "" {
		return "", fmt.Errorf("anon token empty")
	}
	return body.Token, nil
}

func getModelNames() []string {
	names := make([]string, 0, len(MODEL_MAP))
	for name := range MODEL_MAP {
		names = append(names, name)
	}
	return names
}

func main() {
	// 初始化配置
	initConfig()

	http.HandleFunc("/v1/models", handleModels)
	http.HandleFunc("/v1/chat/completions", handleChatCompletions)
	http.HandleFunc("/", handleOptions)

	log.Printf("OpenAI兼容API服务器启动在端口%s", PORT)
	log.Printf("可用模型: %v", getModelNames())
	log.Printf("上游: %s", UPSTREAM_URL)
	log.Printf("Debug模式: %v", DEBUG_MODE)
	log.Printf("默认流式响应: %v", DEFAULT_STREAM)
	log.Fatal(http.ListenAndServe(PORT, nil))
}

func handleOptions(w http.ResponseWriter, r *http.Request) {
	setCORSHeaders(w)
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}
	w.WriteHeader(http.StatusNotFound)
}

func setCORSHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	w.Header().Set("Access-Control-Allow-Credentials", "true")
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	setCORSHeaders(w)
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	models := make([]Model, 0, len(MODEL_MAP))
	for modelName := range MODEL_MAP {
		models = append(models, Model{
			ID:      modelName,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "z.ai",
		})
	}

	response := ModelsResponse{
		Object: "list",
		Data:   models,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	setCORSHeaders(w)
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	debugLog("收到chat completions请求")

	// 验证API Key
	authHeader := r.Header.Get("Authorization")
	if !strings.HasPrefix(authHeader, "Bearer ") {
		debugLog("缺少或无效的Authorization头")
		http.Error(w, "Missing or invalid Authorization header", http.StatusUnauthorized)
		return
	}

	apiKey := strings.TrimPrefix(authHeader, "Bearer ")
	if apiKey != DEFAULT_KEY {
		debugLog("无效的API key: %s", apiKey)
		http.Error(w, "Invalid API key", http.StatusUnauthorized)
		return
	}

	debugLog("API key验证通过")

	// 读取请求体
	body, err := io.ReadAll(r.Body)
	if err != nil {
		debugLog("读取请求体失败: %v", err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}

	// 解析请求
	var req OpenAIRequest
	if err := json.Unmarshal(body, &req); err != nil {
		debugLog("JSON解析失败: %v", err)
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// 如果客户端没有明确指定stream参数，使用默认值
	if !bytes.Contains(body, []byte(`"stream"`)) {
		req.Stream = DEFAULT_STREAM
		debugLog("客户端未指定stream参数，使用默认值: %v", DEFAULT_STREAM)
	}

	// New model lookup logic
	upstreamModelID, ok := MODEL_MAP[req.Model]
	if !ok {
		debugLog("不支持的模型: %s", req.Model)
		http.Error(w, fmt.Sprintf("Unsupported model: %s. Available models: %v", req.Model, getModelNames()), http.StatusBadRequest)
		return
	}

	debugLog("请求解析成功 - display_model: %s, upstream_model: %s, 流式: %v, 消息数: %d", req.Model, upstreamModelID, req.Stream, len(req.Messages))

	// 生成会话相关ID
	chatID := fmt.Sprintf("%d-%d", time.Now().UnixNano(), time.Now().Unix())
	msgID := fmt.Sprintf("%d", time.Now().UnixNano())

	// 构造上游请求
	upstreamReq := UpstreamRequest{
		Stream:   true, // 总是使用流式从上游获取
		ChatID:   chatID,
		ID:       msgID,
		Model:    upstreamModelID,
		Messages: req.Messages,
		Params:   map[string]interface{}{},
		Features: map[string]interface{}{
			"enable_thinking": true,
		},
		BackgroundTasks: map[string]bool{
			"title_generation": false,
			"tags_generation":  false,
		},
		MCPServers: []string{},
		ModelItem: struct {
			ID      string `json:"id"`
			Name    string `json:"name"`
			OwnedBy string `json:"owned_by"`
		}{ID: upstreamModelID, Name: req.Model, OwnedBy: "openai"},
		ToolServers: []string{},
		Variables: map[string]string{
			"{{USER_NAME}}":        "User",
			"{{USER_LOCATION}}":    "Unknown",
			"{{CURRENT_DATETIME}}": time.Now().Format("2006-01-02 15:04:05"),
		},
	}

	// 选择本次对话使用的token
	authToken := UPSTREAM_TOKEN
	if ANON_TOKEN_ENABLED {
		if t, err := getAnonymousToken(); err == nil {
			authToken = t
			debugLog("匿名token获取成功: %s...", func() string {
				if len(t) > 10 {
					return t[:10]
				}
				return t
			}())
		} else {
			debugLog("匿名token获取失败，回退固定token: %v", err)
		}
	}

	// 调用上游API
	if req.Stream {
		handleStreamResponseWithIDs(w, upstreamReq, chatID, authToken, req.Model)
	} else {
		handleNonStreamResponseWithIDs(w, upstreamReq, chatID, authToken, req.Model)
	}
}

func callUpstreamWithHeaders(upstreamReq UpstreamRequest, refererChatID string, authToken string) (*http.Response, error) {
	reqBody, err := json.Marshal(upstreamReq)
	if err != nil {
		debugLog("上游请求序列化失败: %v", err)
		return nil, err
	}

	debugLog("调用上游API: %s", UPSTREAM_URL)
	debugLog("上游请求体: %s", string(reqBody))

	req, err := http.NewRequest("POST", UPSTREAM_URL, bytes.NewBuffer(reqBody))
	if err != nil {
		debugLog("创建HTTP请求失败: %v", err)
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("User-Agent", BROWSER_UA)
	req.Header.Set("Authorization", "Bearer "+authToken)
	req.Header.Set("Accept-Language", "zh-CN")
	req.Header.Set("sec-ch-ua", SEC_CH_UA)
	req.Header.Set("sec-ch-ua-mobile", SEC_CH_UA_MOB)
	req.Header.Set("sec-ch-ua-platform", SEC_CH_UA_PLAT)
	req.Header.Set("X-FE-Version", X_FE_VERSION)
	req.Header.Set("Origin", ORIGIN_BASE)
	req.Header.Set("Referer", ORIGIN_BASE+"/c/"+refererChatID)

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		debugLog("上游请求失败: %v", err)
		return nil, err
	}

	debugLog("上游响应状态: %d %s", resp.StatusCode, resp.Status)
	return resp, nil
}

func handleStreamResponseWithIDs(w http.ResponseWriter, upstreamReq UpstreamRequest, chatID string, authToken string, displayModel string) {
	debugLog("开始处理流式响应 (chat_id=%s)", chatID)

	resp, err := callUpstreamWithHeaders(upstreamReq, chatID, authToken)
	if err != nil {
		debugLog("调用上游失败: %v", err)
		http.Error(w, "Failed to call upstream", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		debugLog("上游返回错误状态: %d", resp.StatusCode)
		// 读取错误响应体
		if DEBUG_MODE {
			body, _ := io.ReadAll(resp.Body)
			debugLog("上游错误响应: %s", string(body))
		}
		http.Error(w, "Upstream error", http.StatusBadGateway)
		return
	}

	// 用于策略2：总是展示thinking（配合标签处理）
	transformThinking := func(s string) string {
		// 去 <summary>…</summary>
		s = regexp.MustCompile(`(?s)<summary>.*?</summary>`).ReplaceAllString(s, "")
		// 清理残留自定义标签，如 </thinking>、<Full> 等
		s = strings.ReplaceAll(s, "</thinking>", "")
		s = strings.ReplaceAll(s, "<Full>", "")
		s = strings.ReplaceAll(s, "</Full>", "")
		s = strings.TrimSpace(s)
		switch THINK_TAGS_MODE {
		case "think":
			s = regexp.MustCompile(`<details[^>]*>`).ReplaceAllString(s, "<think>")
			s = strings.ReplaceAll(s, "</details>", "</think>")
		case "strip":
			s = regexp.MustCompile(`<details[^>]*>`).ReplaceAllString(s, "")
			s = strings.ReplaceAll(s, "</details>", "")
		}
		// 处理每行前缀 "> "（包括起始位置）
		s = strings.TrimPrefix(s, "> ")
		s = strings.ReplaceAll(s, "\n> ", "\n")
		return strings.TrimSpace(s)
	}

	// 设置SSE头部
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	// 发送第一个chunk（role）
	firstChunk := OpenAIResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   displayModel,
		Choices: []Choice{
			{
				Index: 0,
				Delta: Delta{Role: "assistant"},
			},
		},
	}
	writeSSEChunk(w, firstChunk)
	flusher.Flush()

	// 读取上游SSE流
	debugLog("开始读取上游SSE流")
	scanner := bufio.NewScanner(resp.Body)
	lineCount := 0

	for scanner.Scan() {
		line := scanner.Text()
		lineCount++

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		dataStr := strings.TrimPrefix(line, "data: ")
		if dataStr == "" {
			continue
		}

		debugLog("收到SSE数据 (第%d行): %s", lineCount, dataStr)

		var upstreamData UpstreamData
		if err := json.Unmarshal([]byte(dataStr), &upstreamData); err != nil {
			debugLog("SSE数据解析失败: %v", err)
			continue
		}

		// 错误检测（data.error 或 data.data.error 或 顶层error）
		if (upstreamData.Error != nil) || (upstreamData.Data.Error != nil) || (upstreamData.Data.Inner != nil && upstreamData.Data.Inner.Error != nil) {
			errObj := upstreamData.Error
			if errObj == nil {
				errObj = upstreamData.Data.Error
			}
			if errObj == nil && upstreamData.Data.Inner != nil {
				errObj = upstreamData.Data.Inner.Error
			}
			debugLog("上游错误: code=%d, detail=%s", errObj.Code, errObj.Detail)
			// 结束下游流
			endChunk := OpenAIResponse{
				ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   displayModel,
				Choices: []Choice{{Index: 0, Delta: Delta{}, FinishReason: "stop"}},
			}
			writeSSEChunk(w, endChunk)
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			break
		}

		debugLog("解析成功 - 类型: %s, 阶段: %s, 内容长度: %d, 完成: %v",
			upstreamData.Type, upstreamData.Data.Phase, len(upstreamData.Data.DeltaContent), upstreamData.Data.Done)

		// 策略2：总是展示thinking + answer
		if upstreamData.Data.DeltaContent != "" {
			var out = upstreamData.Data.DeltaContent
			if upstreamData.Data.Phase == "thinking" {
				out = transformThinking(out)
			}
			if out != "" {
				debugLog("发送内容(%s): %s", upstreamData.Data.Phase, out)
				chunk := OpenAIResponse{
					ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
					Object:  "chat.completion.chunk",
					Created: time.Now().Unix(),
					Model:   displayModel,
					Choices: []Choice{
						{
							Index: 0,
							Delta: Delta{Content: out},
						},
					},
				}
				writeSSEChunk(w, chunk)
				flusher.Flush()
			}
		}

		// 检查是否结束
		if upstreamData.Data.Done || upstreamData.Data.Phase == "done" {
			debugLog("检测到流结束信号")
			// 发送结束chunk
			endChunk := OpenAIResponse{
				ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   displayModel,
				Choices: []Choice{
					{
						Index:        0,
						Delta:        Delta{},
						FinishReason: "stop",
					},
				},
			}
			writeSSEChunk(w, endChunk)
			flusher.Flush()

			// 发送[DONE]
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			debugLog("流式响应完成，共处理%d行", lineCount)
			break
		}
	}

	if err := scanner.Err(); err != nil {
		debugLog("扫描器错误: %v", err)
	}
}

func writeSSEChunk(w http.ResponseWriter, chunk OpenAIResponse) {
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
}

func handleNonStreamResponseWithIDs(w http.ResponseWriter, upstreamReq UpstreamRequest, chatID string, authToken string, displayModel string) {
	debugLog("开始处理非流式响应 (chat_id=%s)", chatID)

	resp, err := callUpstreamWithHeaders(upstreamReq, chatID, authToken)
	if err != nil {
		debugLog("调用上游失败: %v", err)
		http.Error(w, "Failed to call upstream", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		debugLog("上游返回错误状态: %d", resp.StatusCode)
		// 读取错误响应体
		if DEBUG_MODE {
			body, _ := io.ReadAll(resp.Body)
			debugLog("上游错误响应: %s", string(body))
		}
		http.Error(w, "Upstream error", http.StatusBadGateway)
		return
	}

	// 收集完整响应（策略2：thinking与answer都纳入，thinking转换）
	var fullContent strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	debugLog("开始收集完整响应内容")

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		dataStr := strings.TrimPrefix(line, "data: ")
		if dataStr == "" {
			continue
		}

		var upstreamData UpstreamData
		if err := json.Unmarshal([]byte(dataStr), &upstreamData); err != nil {
			continue
		}

		if upstreamData.Data.DeltaContent != "" {
			out := upstreamData.Data.DeltaContent
			if upstreamData.Data.Phase == "thinking" {
				out = func(s string) string {
					// 同步一份转换逻辑（与流式一致）
					s = regexp.MustCompile(`(?s)<summary>.*?</summary>`).ReplaceAllString(s, "")
					s = strings.ReplaceAll(s, "</thinking>", "")
					s = strings.ReplaceAll(s, "<Full>", "")
					s = strings.ReplaceAll(s, "</Full>", "")
					s = strings.TrimSpace(s)
					switch THINK_TAGS_MODE {
					case "think":
						s = regexp.MustCompile(`<details[^>]*>`).ReplaceAllString(s, "<think>")
						s = strings.ReplaceAll(s, "</details>", "</think>")
					case "strip":
						s = regexp.MustCompile(`<details[^>]*>`).ReplaceAllString(s, "")
						s = strings.ReplaceAll(s, "</details>", "")
					}
					s = strings.TrimPrefix(s, "> ")
					s = strings.ReplaceAll(s, "\n> ", "\n")
					return strings.TrimSpace(s)
				}(out)
			}
			if out != "" {
				fullContent.WriteString(out)
			}
		}

		if upstreamData.Data.Done || upstreamData.Data.Phase == "done" {
			debugLog("检测到完成信号，停止收集")
			break
		}
	}

	finalContent := fullContent.String()
	debugLog("内容收集完成，最终长度: %d", len(finalContent))

	// 构造完整响应
	response := OpenAIResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   displayModel,
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: finalContent,
				},
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			PromptTokens:     0,
			CompletionTokens: 0,
			TotalTokens:      0,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
	debugLog("非流式响应发送完成")
}
