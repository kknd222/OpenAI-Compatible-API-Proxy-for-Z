package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// Config variables from environment
var (
	UPSTREAM_URL   string
	DEFAULT_KEY    string
	UPSTREAM_TOKEN string
	MODEL_MAP      map[string]string
	PORT           string
	DEBUG_MODE     bool
	DEFAULT_STREAM bool
)

// Constants
const (
	X_FE_VERSION     = "prod-fe-1.0.70"
	BROWSER_UA       = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0"
	ORIGIN_BASE      = "https://chat.z.ai"
	ANON_TOKEN_ENABLED = true
	THINK_TAGS_MODE = "strip"
)

// Init config from environment variables
func initConfig() {
	UPSTREAM_URL = getEnv("UPSTREAM_URL", "https://chat.z.ai/api/chat/completions")
	DEFAULT_KEY = getEnv("DEFAULT_KEY", "sk-your-key")
	UPSTREAM_TOKEN = getEnv("UPSTREAM_TOKEN", "") // Must be set by user
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

	if !strings.HasPrefix(PORT, ":") {
		PORT = ":" + PORT
	}
	DEBUG_MODE = getEnv("DEBUG_MODE", "true") == "true"
	DEFAULT_STREAM = getEnv("DEFAULT_STREAM", "true") == "true"
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// Structs
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

type UpstreamRequest struct {
	Stream          bool                   `json:"stream"`
	Model           string                 `json:"model"`
	Messages        []Message              `json:"messages"`
	Params          map[string]interface{} `json:"params"`
	Features        map[string]interface{} `json:"features"`
	BackgroundTasks map[string]bool        `json:"background_tasks,omitempty"`
	ChatID          string                 `json:"chat_id,omitempty"`
	ID              string                 `json:"id,omitempty"`
	ModelItem       struct {
		ID      string `json:"id"`
		Name    string `json:"name"`
		OwnedBy string `json:"owned_by"`
	} `json:"model_item,omitempty"`
}

type OpenAIResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
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

func debugLog(format string, args ...interface{}) {
	if DEBUG_MODE {
		log.Printf("[DEBUG] "+format, args...)
	}
}

func getModelNames() []string {
	names := make([]string, 0, len(MODEL_MAP))
	for name := range MODEL_MAP {
		names = append(names, name)
	}
	return names
}

func getAnonymousToken() (string, error) {
	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest("GET", ORIGIN_BASE+"/api/v1/auths/", nil)
	if err != nil { return "", err }
	req.Header.Set("User-Agent", BROWSER_UA)
	req.Header.Set("Accept", "*/*")
	req.Header.Set("Origin", ORIGIN_BASE)
	req.Header.Set("Referer", ORIGIN_BASE+"/")
	resp, err := client.Do(req)
	if err != nil { return "", err }
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK { return "", fmt.Errorf("anon token status=%d", resp.StatusCode) }
	var body struct { Token string `json:"token"` }
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil { return "", err }
	if body.Token == "" { return "", fmt.Errorf("anon token empty") }
	return body.Token, nil
}

func main() {
	initConfig()
	http.HandleFunc("/v1/models", handleModels)
	http.HandleFunc("/v1/chat/completions", handleChatCompletions)
	http.HandleFunc("/", handleOptions)
	log.Printf("Server starting on port %s", PORT)
	log.Printf("Upstream: %s", UPSTREAM_URL)
	log.Printf("Supported Models: %v", getModelNames())
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
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	setCORSHeaders(w)
	var models []Model
	for name := range MODEL_MAP {
		models = append(models, Model{ID: name, Object: "model", Created: time.Now().Unix(), OwnedBy: "z.ai"})
	}
	json.NewEncoder(w).Encode(ModelsResponse{Object: "list", Data: models})
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	setCORSHeaders(w)

	// Auth check
	if strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ") != DEFAULT_KEY {
		http.Error(w, "Invalid API key", http.StatusUnauthorized)
		return
	}

	// Read and parse request
	var req OpenAIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Get upstream model ID
	upstreamModelID, ok := MODEL_MAP[req.Model]
	if !ok {
		http.Error(w, "Unsupported model", http.StatusBadRequest)
		return
	}

	// Get auth token
	authToken := UPSTREAM_TOKEN
	if ANON_TOKEN_ENABLED {
		if t, err := getAnonymousToken(); err == nil {
			authToken = t
		}
	}

	// Build upstream request
	chatID := fmt.Sprintf("%d-%d", time.Now().UnixNano(), time.Now().Unix())
	upstreamReq := UpstreamRequest{
		Stream:   true,
		Model:    upstreamModelID,
		Messages: req.Messages,
		Params:   map[string]interface{}{},
		Features: map[string]interface{}{"enable_thinking": true},
		ChatID:   chatID,
		ModelItem: struct {
			ID      string `json:"id"`
			Name    string `json:"name"`
			OwnedBy string `json:"owned_by"`
		}{ID: upstreamModelID, Name: req.Model, OwnedBy: "openai"},
	}

	// Make the request
	upstreamResp, err := callUpstream(upstreamReq, chatID, authToken)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer upstreamResp.Body.Close()

	// Proxy the response
	for h, val := range upstreamResp.Header {
		w.Header()[h] = val
	}
	w.WriteHeader(upstreamResp.StatusCode)
	io.Copy(w, upstreamResp.Body)
}

func callUpstream(upstreamReq UpstreamRequest, refererChatID string, authToken string) (*http.Response, error) {
	reqBody, err := json.Marshal(upstreamReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal upstream request: %v", err)
	}

	req, err := http.NewRequest("POST", UPSTREAM_URL, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create upstream request: %v", err)
	}

	req.Header.Set("Authorization", "Bearer "+authToken)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("User-Agent", BROWSER_UA)
	req.Header.Set("Origin", ORIGIN_BASE)
	req.Header.Set("Referer", ORIGIN_BASE+"/c/"+refererChatID)

	client := &http.Client{Timeout: 60 * time.Second}
	return client.Do(req)
}
