use std::{
    env,
    net::SocketAddr,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, HeaderMap, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;
use tracing::error;
use uuid::Uuid;
use whatlang::{detect, Lang};

#[derive(Clone)]
struct AppState {
    api_key: Option<String>,
    default_to: String,
    model_name: String,
    llama_base_url: String,
    http: Client,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    fn upstream(status: StatusCode, msg: impl Into<String>) -> Self {
        Self {
            status,
            message: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

#[derive(Deserialize)]
struct StreamQuery {
    #[serde(default)]
    stream: Option<String>,
}

fn stream_enabled(q: &StreamQuery) -> bool {
    matches!(q.stream.as_deref(), Some("1" | "true" | "yes" | "on"))
}

async fn auth_middleware(
    State(state): State<AppState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let Some(expected) = state.api_key.as_deref().filter(|s| !s.is_empty()) else {
        return Ok(next.run(req).await);
    };

    let headers = req.headers();
    let auth = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok());
    let bearer_ok = auth
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.trim() == expected)
        .unwrap_or(false);

    let query_ok = req
        .uri()
        .query()
        .and_then(|q| {
            for kv in q.split('&') {
                let mut it = kv.splitn(2, '=');
                let k = it.next()?;
                let v = it.next()?;
                if k == "token" {
                    return Some(v == expected);
                }
            }
            None
        })
        .unwrap_or(false);

    if bearer_ok || query_ok {
        Ok(next.run(req).await)
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

fn normalize_lang(code: Option<&str>) -> Option<String> {
    let c = code?.trim();
    if c.is_empty() {
        return None;
    }
    let c_lower = c.to_lowercase();
    if matches!(c_lower.as_str(), "auto" | "detect") {
        return None;
    }
    if matches!(c_lower.as_str(), "zh-cn" | "zh-hans") {
        return Some("zh".to_string());
    }
    if matches!(c_lower.as_str(), "zh-tw" | "zh-hk" | "zh-hant") {
        return Some("zh-Hant".to_string());
    }
    if c_lower == "jp" {
        return Some("ja".to_string());
    }
    if c_lower == "cn" {
        return Some("zh".to_string());
    }

    let mut out = String::new();
    for ch in c_lower.chars() {
        if ch.is_ascii_alphabetic() {
            out.push(ch);
            if out.len() >= 3 {
                break;
            }
        } else {
            break;
        }
    }
    if out.len() >= 2 {
        Some(out)
    } else {
        Some(c.to_string())
    }
}

fn detect_language_code(text: &str) -> String {
    let t = text.trim();
    if t.is_empty() {
        return "en".to_string();
    }

    // Fast, deterministic script heuristics first (avoid short-text mis-detection).
    let mut has_han = false;
    let mut has_kana = false;
    let mut has_hangul = false;
    for ch in t.chars() {
        if ('\u{AC00}'..='\u{D7AF}').contains(&ch) {
            has_hangul = true;
        } else if ('\u{3040}'..='\u{30FF}').contains(&ch) {
            has_kana = true;
        } else if ('\u{3400}'..='\u{4DBF}').contains(&ch) || ('\u{4E00}'..='\u{9FFF}').contains(&ch) {
            has_han = true;
        }
    }
    if has_hangul {
        return "ko".to_string();
    }
    if has_kana {
        return "ja".to_string();
    }
    if has_han {
        return "zh".to_string();
    }

    // If it's short and overwhelmingly ASCII, default to English.
    let total = t.chars().count();
    if total > 0 && total <= 32 {
        let ascii_total = t.chars().filter(|c| c.is_ascii()).count();
        let ascii_letters = t.chars().filter(|c| c.is_ascii_alphabetic()).count();
        if ascii_letters > 0 && ascii_total.saturating_mul(100) / total >= 90 {
            return "en".to_string();
        }
    }

    let Some(info) = detect(t) else {
        return "en".to_string();
    };

    // whatlang can be wrong on short/ambiguous strings; require high confidence.
    if !info.is_reliable() && info.confidence() < 0.90 {
        return "en".to_string();
    }

    match info.lang() {
        Lang::Eng => "en",
        Lang::Cmn => "zh",
        Lang::Jpn => "ja",
        Lang::Kor => "ko",
        Lang::Rus => "ru",
        Lang::Fra => "fr",
        Lang::Deu => "de",
        Lang::Spa => "es",
        Lang::Por => "pt",
        Lang::Ita => "it",
        Lang::Tur => "tr",
        Lang::Ara => "ar",
        Lang::Hin => "hi",
        Lang::Ukr => "uk",
        Lang::Nld => "nl",
        Lang::Ces => "cs",
        Lang::Pol => "pl",
        Lang::Vie => "vi",
        Lang::Ind => "id",
        Lang::Tha => "th",
        Lang::Heb => "he",
        _ => info.lang().code(),
    }
    .to_string()
}

fn target_language_name(code: &str) -> &str {
    match code {
        "zh" => "Chinese",
        "zh-Hant" => "Traditional Chinese",
        "en" => "English",
        "fr" => "French",
        "pt" => "Portuguese",
        "es" => "Spanish",
        "ja" => "Japanese",
        "tr" => "Turkish",
        "ru" => "Russian",
        "ar" => "Arabic",
        "ko" => "Korean",
        "th" => "Thai",
        "it" => "Italian",
        "de" => "German",
        "vi" => "Vietnamese",
        "id" => "Indonesian",
        "hi" => "Hindi",
        "pl" => "Polish",
        "cs" => "Czech",
        "nl" => "Dutch",
        "he" => "Hebrew",
        _ => code,
    }
}

fn build_translation_prompt(text: &str, to_lang: &str) -> String {
    let lang_name = target_language_name(to_lang);
    format!(
        "Translate the following segment into {lang_name}, without additional explanation.\n\n{text}"
    )
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_string(name: &str) -> Option<String> {
    let v = env::var(name).ok()?;
    let t = v.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

async fn llama_chat_completion_json(
    state: &AppState,
    prompt: String,
    stream: bool,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> Result<Response, ApiError> {
    let url = format!("{}/v1/chat/completions", state.llama_base_url.trim_end_matches('/'));

    let payload = json!({
        "model": state.model_name,
        "messages": [{ "role": "user", "content": prompt }],
        "stream": stream,
        "max_tokens": max_tokens as i64,
        "temperature": temperature,
        "top_p": top_p,
    });

    let res = state
        .http
        .post(url)
        .header(header::ACCEPT, if stream { "text/event-stream" } else { "application/json" })
        .json(&payload)
        .send()
        .await
        .map_err(|e| ApiError::upstream(StatusCode::BAD_GATEWAY, format!("llama-server request failed: {e}")))?;

    let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    if !res.status().is_success() {
        let text = res.text().await.unwrap_or_default();
        return Err(ApiError::upstream(status, format!("llama-server error: {text}")));
    }

    if stream {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );
        headers.insert(header::CACHE_CONTROL, header::HeaderValue::from_static("no-cache"));

        let stream = res.bytes_stream();
        let body = Body::from_stream(stream);
        let mut out = Response::new(body);
        *out.status_mut() = StatusCode::OK;
        *out.headers_mut() = headers;
        Ok(out)
    } else {
        let v: Value = res
            .json()
            .await
            .map_err(|e| ApiError::upstream(StatusCode::BAD_GATEWAY, format!("invalid llama-server JSON: {e}")))?;
        Ok(Json(v).into_response())
    }
}

async fn llama_chat_content(
    state: &AppState,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> Result<String, ApiError> {
    let url = format!("{}/v1/chat/completions", state.llama_base_url.trim_end_matches('/'));
    let payload = json!({
        "model": state.model_name,
        "messages": [{ "role": "user", "content": prompt }],
        "stream": false,
        "max_tokens": max_tokens as i64,
        "temperature": temperature,
        "top_p": top_p,
    });
    let res = state
        .http
        .post(url)
        .json(&payload)
        .send()
        .await
        .map_err(|e| ApiError::upstream(StatusCode::BAD_GATEWAY, format!("llama-server request failed: {e}")))?;

    let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    if !res.status().is_success() {
        let text = res.text().await.unwrap_or_default();
        return Err(ApiError::upstream(status, format!("llama-server error: {text}")));
    }

    let v: Value = res
        .json()
        .await
        .map_err(|e| ApiError::upstream(StatusCode::BAD_GATEWAY, format!("invalid llama-server JSON: {e}")))?;
    Ok(v
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c0| c0.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .trim()
        .to_string())
}

async fn health_handler() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

async fn detect_handler(Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let text = body
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("missing field: text"))?;
    Ok(Json(json!({ "language": detect_language_code(text) })))
}

async fn translate_handler(
    State(state): State<AppState>,
    Query(q): Query<StreamQuery>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let text = body
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("missing field: text"))?
        .to_string();
    let to_lang = normalize_lang(body.get("to").and_then(|v| v.as_str()))
        .unwrap_or_else(|| "en".to_string());
    let from_lang = normalize_lang(
        body.get("from")
            .or_else(|| body.get("from_lang"))
            .and_then(|v| v.as_str()),
    );
    let detected = from_lang.unwrap_or_else(|| detect_language_code(&text));

    let max_tokens = env_usize("MAX_NEW_TOKENS", 1024);
    let temperature = env_f32("TEMPERATURE", 0.0);
    let top_p = env_f32("TOP_P", 0.6);

    let prompt = build_translation_prompt(&text, &to_lang);

    if stream_enabled(&q) {
        llama_chat_completion_json(&state, prompt, true, max_tokens, temperature, top_p).await
    } else {
        let content = llama_chat_content(&state, prompt, max_tokens, temperature, top_p).await?;
        Ok(Json(json!({ "text": content, "from": detected, "to": to_lang })).into_response())
    }
}

async fn kiss_handler(
    State(state): State<AppState>,
    Query(q): Query<StreamQuery>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    translate_handler(State(state), Query(q), Json(body)).await
}

async fn imme_handler(
    State(state): State<AppState>,
    Query(q): Query<StreamQuery>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let text_list = body
        .get("text_list")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ApiError::bad_request("missing field: text_list"))?;
    let target_lang = normalize_lang(body.get("target_lang").and_then(|v| v.as_str()))
        .unwrap_or_else(|| "en".to_string());
    let source_lang = normalize_lang(body.get("source_lang").and_then(|v| v.as_str()));

    let max_texts = env_usize("IMME_MAX_TEXTS", 1024);
    if max_texts > 0 && text_list.len() > max_texts {
        return Err(ApiError::upstream(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("text_list too large ({}); limit is IMME_MAX_TEXTS={max_texts}", text_list.len()),
        ));
    }

    let max_tokens = env_usize("MAX_NEW_TOKENS", 1024);
    let temperature = env_f32("TEMPERATURE", 0.0);
    let top_p = env_f32("TOP_P", 0.6);

    if stream_enabled(&q) {
        // Basic streaming: return one JSON item per SSE delta (not token-level).
        // For token-level streaming, clients should use /v1/chat/completions directly.
        let id = format!("chatcmpl-{}", Uuid::new_v4());
        let created = now_unix_seconds();
        let model_name = state.model_name.clone();

        let mut events: Vec<String> = Vec::with_capacity(text_list.len() + 3);
        events.push(
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": { "role": "assistant" },
                    "finish_reason": Value::Null
                }],
            })
            .to_string(),
        );

        for v in text_list {
            let t = v.as_str().unwrap_or_default().to_string();
            let detected = source_lang.clone().unwrap_or_else(|| detect_language_code(&t));
            let prompt = build_translation_prompt(&t, &target_lang);
            let url = format!("{}/v1/chat/completions", state.llama_base_url.trim_end_matches('/'));
            let payload = json!({
                "model": state.model_name,
                "messages": [{ "role": "user", "content": prompt }],
                "stream": false,
                "max_tokens": max_tokens as i64,
                "temperature": temperature,
                "top_p": top_p,
            });
            let res = state
                .http
                .post(url)
                .json(&payload)
                .send()
                .await
                .map_err(|e| ApiError::upstream(StatusCode::BAD_GATEWAY, format!("llama-server request failed: {e}")))?;
            let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            if !res.status().is_success() {
                let text = res.text().await.unwrap_or_default();
                return Err(ApiError::upstream(status, format!("llama-server error: {text}")));
            }
            let j: Value = res
                .json()
                .await
                .map_err(|e| ApiError::upstream(StatusCode::BAD_GATEWAY, format!("invalid llama-server JSON: {e}")))?;
            let out = j
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c0| c0.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let item = json!({ "detected_source_lang": detected, "text": out });
            let ev = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": { "content": item.to_string() },
                    "finish_reason": Value::Null
                }],
            })
            .to_string();
            events.push(ev);
        }

        events.push(
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }],
            })
            .to_string(),
        );
        events.push("[DONE]".to_string());

        let stream = tokio_stream::iter(events.into_iter().map(|data| {
            Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(data))
        }));
        Ok(axum::response::Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response())
    } else {
        let mut out_items: Vec<Value> = Vec::with_capacity(text_list.len());
        for v in text_list {
            let t = v.as_str().unwrap_or_default().to_string();
            let detected = source_lang.clone().unwrap_or_else(|| detect_language_code(&t));
            let prompt = build_translation_prompt(&t, &target_lang);
            let url = format!("{}/v1/chat/completions", state.llama_base_url.trim_end_matches('/'));
            let payload = json!({
                "model": state.model_name,
                "messages": [{ "role": "user", "content": prompt }],
                "stream": false,
                "max_tokens": max_tokens as i64,
                "temperature": temperature,
                "top_p": top_p,
            });
            let res = state
                .http
                .post(url)
                .json(&payload)
                .send()
                .await
                .map_err(|e| ApiError::upstream(StatusCode::BAD_GATEWAY, format!("llama-server request failed: {e}")))?;
            let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            if !res.status().is_success() {
                let text = res.text().await.unwrap_or_default();
                return Err(ApiError::upstream(status, format!("llama-server error: {text}")));
            }
            let j: Value = res
                .json()
                .await
                .map_err(|e| ApiError::upstream(StatusCode::BAD_GATEWAY, format!("invalid llama-server JSON: {e}")))?;
            let out = j
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c0| c0.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            out_items.push(json!({ "detected_source_lang": detected, "text": out }));
        }
        Ok(Json(json!({ "translations": out_items })).into_response())
    }
}

async fn openai_models_handler(State(state): State<AppState>) -> impl IntoResponse {
    let created = now_unix_seconds();
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": created,
            "owned_by": "local"
        }]
    }))
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    to: Option<String>,
    #[serde(default)]
    from: Option<String>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

async fn openai_chat_completions_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let to_lang = req
        .to
        .as_deref()
        .and_then(|s| normalize_lang(Some(s)))
        .unwrap_or_else(|| state.default_to.clone());
    let from_lang = req.from.as_deref().and_then(|s| normalize_lang(Some(s)));

    let model_name = req.model.clone().unwrap_or_else(|| state.model_name.clone());
    let user_text = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_else(|| {
            req.messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        });

    let _detected = from_lang.unwrap_or_else(|| detect_language_code(&user_text));
    let prompt = build_translation_prompt(&user_text, &to_lang);

    let max_tokens = req.max_tokens.unwrap_or_else(|| env_usize("MAX_NEW_TOKENS", 1024));
    let temperature = req.temperature.unwrap_or_else(|| env_f32("TEMPERATURE", 0.0));
    let top_p = req.top_p.unwrap_or_else(|| env_f32("TOP_P", 0.6));

    let stream = req.stream.unwrap_or(false);

    // If streaming, proxy upstream SSE directly for true token streaming.
    if stream {
        let url = format!("{}/v1/chat/completions", state.llama_base_url.trim_end_matches('/'));
        let payload = json!({
            "model": model_name,
            "messages": [{ "role": "user", "content": prompt }],
            "stream": true,
            "max_tokens": max_tokens as i64,
            "temperature": temperature,
            "top_p": top_p,
        });
        let res = state
            .http
            .post(url)
            .header(header::ACCEPT, "text/event-stream")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ApiError::upstream(StatusCode::BAD_GATEWAY, format!("llama-server request failed: {e}")))?;
        let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        if !res.status().is_success() {
            let text = res.text().await.unwrap_or_default();
            return Err(ApiError::upstream(status, format!("llama-server error: {text}")));
        }

        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );
        headers.insert(header::CACHE_CONTROL, header::HeaderValue::from_static("no-cache"));
        let body = Body::from_stream(res.bytes_stream());
        let mut out = Response::new(body);
        *out.status_mut() = StatusCode::OK;
        *out.headers_mut() = headers;
        Ok(out)
    } else {
        let url = format!("{}/v1/chat/completions", state.llama_base_url.trim_end_matches('/'));
        let payload = json!({
            "model": model_name,
            "messages": [{ "role": "user", "content": prompt }],
            "stream": false,
            "max_tokens": max_tokens as i64,
            "temperature": temperature,
            "top_p": top_p,
        });
        let res = state
            .http
            .post(url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ApiError::upstream(StatusCode::BAD_GATEWAY, format!("llama-server request failed: {e}")))?;
        let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        if !res.status().is_success() {
            let text = res.text().await.unwrap_or_default();
            return Err(ApiError::upstream(status, format!("llama-server error: {text}")));
        }
        let v: Value = res
            .json()
            .await
            .map_err(|e| ApiError::upstream(StatusCode::BAD_GATEWAY, format!("invalid llama-server JSON: {e}")))?;
        Ok(Json(v).into_response())
    }
}

fn load_config() -> Result<(String, String, String, Option<String>, String), String> {
    let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = env::var("PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(3000);
    let addr = format!("{host}:{port}");

    let api_key = env_string("API_KEY");
    let default_to = env::var("DEFAULT_TARGET_LANG").unwrap_or_else(|_| "zh".to_string());
    let model_name = env::var("OPENAI_MODEL_NAME").unwrap_or_else(|_| "hy-mt-gguf".to_string());

    let llama_base_url = env::var("LLAMA_SERVER_URL").unwrap_or_else(|_| {
        let p = env::var("LLAMA_SERVER_PORT").ok().and_then(|v| v.parse::<u16>().ok()).unwrap_or(18080);
        format!("http://127.0.0.1:{p}")
    });

    Ok((addr, default_to, model_name, api_key, llama_base_url))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let (addr, default_to, model_name, api_key, llama_base_url) = match load_config() {
        Ok(c) => c,
        Err(e) => {
            error!("{e}");
            std::process::exit(1);
        }
    };

    let state = AppState {
        api_key,
        default_to,
        model_name,
        llama_base_url,
        http: Client::new(),
    };

    let authed = Router::new()
        .route("/detect", post(detect_handler))
        .route("/translate", post(translate_handler))
        .route("/kiss", post(kiss_handler))
        .route("/imme", post(imme_handler))
        .route("/v1/models", get(openai_models_handler))
        .route("/v1/chat/completions", post(openai_chat_completions_handler))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    let app = Router::new()
        .route("/health", get(health_handler))
        .merge(authed)
        .layer(CorsLayer::permissive())
        .with_state(state);

    let sock: SocketAddr = addr.parse().unwrap();
    let listener = tokio::net::TcpListener::bind(sock).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
