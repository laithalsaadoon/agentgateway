use std::io;

use agent_core::strng;
use bytes::Bytes;
use http::HeaderMap;
use http_body_util::BodyExt;
use serde_json::json;

use super::*;
use crate::bedrock::Provider;
use crate::types;

#[tokio::test]
async fn test_append_done_on_success_omits_done_after_error() {
	let mut body = crate::parse::sse::append_done_on_success(axum_core::body::Body::from_stream(
		futures_util::stream::iter(vec![
			Ok::<_, axum_core::Error>(Bytes::from_static(b"data: chunk\n\n")),
			Err(axum_core::Error::new(io::Error::other("boom"))),
		]),
	));

	let first = body
		.frame()
		.await
		.expect("first frame should be present")
		.expect("first frame should succeed")
		.into_data()
		.expect("first frame should contain data");
	assert_eq!(first, Bytes::from_static(b"data: chunk\n\n"));

	let second = body.frame().await.expect("error frame should be present");
	assert!(second.is_err(), "upstream error should be forwarded");
	assert!(
		body.frame().await.is_none(),
		"stream must terminate after an upstream error without appending [DONE]"
	);
}

#[tokio::test]
async fn test_append_done_on_success_does_not_repoll_after_eof() {
	let mut body = crate::parse::sse::append_done_on_success(axum_core::body::Body::from_stream(
		futures_util::stream::iter(vec![Ok::<_, axum_core::Error>(Bytes::from_static(
			b"data: chunk\n\n",
		))]),
	));

	assert!(body.frame().await.is_some(), "data frame should be present");
	assert!(
		body.frame().await.is_some(),
		"[DONE] frame should be present"
	);
	assert!(body.frame().await.is_none(), "stream should report EOF");
	assert!(body.frame().await.is_none(), "stream must remain at EOF");
}

#[test]
fn test_extract_beta_headers_variants() {
	let headers = HeaderMap::new();
	assert!(helpers::extract_beta_headers(&headers).unwrap().is_none());

	let mut headers = HeaderMap::new();
	headers.insert("anthropic-beta", "computer-use-2025-01-24".parse().unwrap());
	assert_eq!(
		helpers::extract_beta_headers(&headers).unwrap().unwrap(),
		vec![json!("computer-use-2025-01-24")]
	);

	let mut headers = HeaderMap::new();
	headers.insert(
		"anthropic-beta",
		"cache-control-2024-08-15,computer-use-2025-01-24,tool-examples-2025-10-29"
			.parse()
			.unwrap(),
	);
	assert_eq!(
		helpers::extract_beta_headers(&headers).unwrap().unwrap(),
		vec![
			json!("computer-use-2025-01-24"),
			json!("tool-examples-2025-10-29"),
		]
	);

	let mut headers = HeaderMap::new();
	headers.insert(
		"anthropic-beta",
		" cache-control-2024-08-15 , computer-use-2025-01-24 "
			.parse()
			.unwrap(),
	);
	assert_eq!(
		helpers::extract_beta_headers(&headers).unwrap().unwrap(),
		vec![json!("computer-use-2025-01-24"),]
	);

	let mut headers = HeaderMap::new();
	headers.append(
		"anthropic-beta",
		"cache-control-2024-08-15".parse().unwrap(),
	);
	headers.append(
		"anthropic-beta",
		"interleaved-thinking-2025-05-14".parse().unwrap(),
	);
	headers.append(
		"anthropic-beta",
		"tool-search-tool-2025-10-19".parse().unwrap(),
	);
	let mut beta_features = helpers::extract_beta_headers(&headers)
		.unwrap()
		.unwrap()
		.into_iter()
		.map(|v| v.as_str().unwrap().to_string())
		.collect::<Vec<_>>();
	beta_features.sort();
	assert_eq!(
		beta_features,
		vec![
			"interleaved-thinking-2025-05-14".to_string(),
			"tool-search-tool-2025-10-19".to_string(),
		]
	);

	let mut headers = HeaderMap::new();
	headers.insert(
		"anthropic-beta",
		"prompt-caching-2024-07-31".parse().unwrap(),
	);
	assert!(helpers::extract_beta_headers(&headers).unwrap().is_none());
}

#[test]
fn test_metadata_from_header() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	// Simulate transformation CEL setting x-bedrock-metadata header
	let mut headers = HeaderMap::new();
	headers.insert(
		"x-bedrock-metadata",
		r#"{"user_id": "user123", "department": "engineering", "json_user": "{\"device_id\":\"abc\"}", "bad?key": "bad{}"}"#
			.parse()
			.unwrap(),
	);

	let req = messages::typed::Request {
		model: "anthropic.claude-3-sonnet".to_string(),
		messages: vec![messages::typed::Message {
			role: messages::typed::Role::User,
			content: vec![messages::typed::ContentBlock::Text(
				messages::typed::ContentTextBlock {
					text: "Hello".to_string(),
					citations: None,
					cache_control: None,
				},
			)],
		}],
		max_tokens: 100,
		metadata: None,
		system: None,
		stop_sequences: vec![],
		stream: false,
		temperature: None,
		top_k: None,
		top_p: None,
		tools: None,
		tool_choice: None,
		thinking: None,
		output_config: None,
	};

	let (out, _) =
		super::from_messages::translate_internal(req, &provider, Some(&headers), None).unwrap();
	let metadata = out.request_metadata.unwrap();

	assert_eq!(metadata.get("user_id"), Some(&"user123".to_string()));
	assert_eq!(metadata.get("department"), Some(&"engineering".to_string()));
	assert_eq!(
		metadata.get("json_user"),
		Some(&r#"{"device_id":"abc"}"#.to_string())
	);
	assert_eq!(metadata.get("bad?key"), Some(&"bad{}".to_string()));
}

#[test]
fn test_output_config_effort_without_thinking_is_passed_through() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = messages::typed::Request {
		model: "anthropic.claude-3-sonnet".to_string(),
		messages: vec![messages::typed::Message {
			role: messages::typed::Role::User,
			content: vec![messages::typed::ContentBlock::Text(
				messages::typed::ContentTextBlock {
					text: "Hello".to_string(),
					citations: None,
					cache_control: None,
				},
			)],
		}],
		max_tokens: 100,
		metadata: None,
		system: None,
		stop_sequences: vec![],
		stream: false,
		temperature: Some(0.7),
		top_k: Some(50),
		top_p: Some(0.8),
		tools: None,
		tool_choice: None,
		thinking: None,
		output_config: Some(messages::typed::OutputConfig {
			effort: Some(messages::typed::ThinkingEffort::High),
			format: None,
		}),
	};

	let (out, _) = super::from_messages::translate_internal(req, &provider, None, None).unwrap();
	assert_eq!(
		out.additional_model_request_fields,
		Some(json!({
			"top_k": 50,
			"output_config": {
				"effort": "high"
			}
		}))
	);
	let inference = out.inference_config.unwrap();
	assert_eq!(inference.temperature, Some(0.7));
	assert_eq!(inference.top_p, Some(0.8));
}

#[test]
fn test_explicit_empty_output_config_is_preserved() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = messages::typed::Request {
		model: "anthropic.claude-3-sonnet".to_string(),
		messages: vec![messages::typed::Message {
			role: messages::typed::Role::User,
			content: vec![messages::typed::ContentBlock::Text(
				messages::typed::ContentTextBlock {
					text: "Hello".to_string(),
					citations: None,
					cache_control: None,
				},
			)],
		}],
		max_tokens: 100,
		metadata: None,
		system: None,
		stop_sequences: vec![],
		stream: false,
		temperature: Some(0.7),
		top_k: Some(50),
		top_p: Some(0.8),
		tools: None,
		tool_choice: None,
		thinking: Some(messages::typed::ThinkingInput::Adaptive {}),
		output_config: Some(messages::typed::OutputConfig {
			effort: None,
			format: None,
		}),
	};

	let (out, _) = super::from_messages::translate_internal(req, &provider, None, None).unwrap();
	assert_eq!(
		out.additional_model_request_fields,
		Some(json!({
			"thinking": {
				"type": "adaptive"
			},
			"top_k": 50,
			"output_config": {}
		}))
	);

	let inference = out.inference_config.unwrap();
	assert_eq!(inference.temperature, Some(0.7));
	assert_eq!(inference.top_p, Some(0.8));
}

#[test]
fn test_thinking_and_output_config_are_both_passed_through() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = messages::typed::Request {
		model: "anthropic.claude-3-sonnet".to_string(),
		messages: vec![messages::typed::Message {
			role: messages::typed::Role::User,
			content: vec![messages::typed::ContentBlock::Text(
				messages::typed::ContentTextBlock {
					text: "Hello".to_string(),
					citations: None,
					cache_control: None,
				},
			)],
		}],
		max_tokens: 100,
		metadata: None,
		system: None,
		stop_sequences: vec![],
		stream: false,
		temperature: None,
		top_k: None,
		top_p: None,
		tools: None,
		tool_choice: None,
		thinking: Some(messages::typed::ThinkingInput::Enabled {
			budget_tokens: 1024,
		}),
		output_config: Some(messages::typed::OutputConfig {
			effort: Some(messages::typed::ThinkingEffort::High),
			format: None,
		}),
	};

	let (out, _) = super::from_messages::translate_internal(req, &provider, None, None).unwrap();
	assert_eq!(
		out.additional_model_request_fields,
		Some(json!({
			"thinking": {
				"type": "enabled",
				"budget_tokens": 1024
			},
			"output_config": {
				"effort": "high"
			}
		}))
	);
}

#[test]
fn test_adaptive_thinking_preserves_sampling_and_tool_choice() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = messages::typed::Request {
		model: "anthropic.claude-3-sonnet".to_string(),
		messages: vec![messages::typed::Message {
			role: messages::typed::Role::User,
			content: vec![messages::typed::ContentBlock::Text(
				messages::typed::ContentTextBlock {
					text: "Hello".to_string(),
					citations: None,
					cache_control: None,
				},
			)],
		}],
		max_tokens: 100,
		metadata: None,
		system: None,
		stop_sequences: vec![],
		stream: false,
		temperature: Some(0.7),
		top_k: Some(50),
		top_p: Some(0.8),
		tools: Some(vec![messages::typed::Tool::Custom(
			messages::typed::CustomTool {
				name: "lookup".to_string(),
				description: Some("Lookup tool".to_string()),
				input_schema: json!({
					"type": "object",
					"properties": {
						"q": { "type": "string" }
					},
					"required": ["q"]
				}),
				cache_control: None,
			},
		)]),
		tool_choice: Some(messages::typed::ToolChoice::Tool {
			name: "lookup".to_string(),
			disable_parallel_tool_use: None,
		}),
		thinking: Some(messages::typed::ThinkingInput::Adaptive {}),
		output_config: None,
	};

	let (out, _) = super::from_messages::translate_internal(req, &provider, None, None).unwrap();
	let inference = out.inference_config.unwrap();
	assert_eq!(inference.temperature, Some(0.7));
	assert_eq!(inference.top_p, Some(0.8));

	let tool_choice = out
		.tool_config
		.as_ref()
		.and_then(|cfg| cfg.tool_choice.as_ref());
	assert!(matches!(
		tool_choice,
		Some(types::bedrock::ToolChoice::Tool { name }) if name == "lookup"
	));

	assert_eq!(
		out.additional_model_request_fields,
		Some(json!({
			"thinking": {
				"type": "adaptive"
			},
			"top_k": 50
		}))
	);
}

#[test]
fn test_enabled_thinking_applies_sampling_and_tool_choice_constraints() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = messages::typed::Request {
		model: "anthropic.claude-3-sonnet".to_string(),
		messages: vec![messages::typed::Message {
			role: messages::typed::Role::User,
			content: vec![messages::typed::ContentBlock::Text(
				messages::typed::ContentTextBlock {
					text: "Hello".to_string(),
					citations: None,
					cache_control: None,
				},
			)],
		}],
		max_tokens: 100,
		metadata: None,
		system: None,
		stop_sequences: vec![],
		stream: false,
		temperature: Some(0.7),
		top_k: Some(50),
		top_p: Some(0.8),
		tools: Some(vec![messages::typed::Tool::Custom(
			messages::typed::CustomTool {
				name: "lookup".to_string(),
				description: Some("Lookup tool".to_string()),
				input_schema: json!({
					"type": "object",
					"properties": {
						"q": { "type": "string" }
					},
					"required": ["q"]
				}),
				cache_control: None,
			},
		)]),
		tool_choice: Some(messages::typed::ToolChoice::Auto {
			disable_parallel_tool_use: None,
		}),
		thinking: Some(messages::typed::ThinkingInput::Enabled {
			budget_tokens: 1024,
		}),
		output_config: None,
	};

	let (out, _) = super::from_messages::translate_internal(req, &provider, None, None).unwrap();
	let inference = out.inference_config.unwrap();
	assert_eq!(inference.temperature, None);
	assert_eq!(inference.top_p, None);

	let tool_choice = out
		.tool_config
		.as_ref()
		.and_then(|cfg| cfg.tool_choice.as_ref());
	assert!(matches!(tool_choice, Some(types::bedrock::ToolChoice::Any)));
}

#[test]
fn test_messages_image_url_to_bedrock_returns_error() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = messages::typed::Request {
		model: "anthropic.claude-3-sonnet".to_string(),
		messages: vec![messages::typed::Message {
			role: messages::typed::Role::User,
			content: vec![messages::typed::ContentBlock::Image(
				messages::typed::ContentImageBlock {
					source: json!({
						"type": "url",
						"url": "https://example.com/sample.jpg"
					}),
					cache_control: None,
				},
			)],
		}],
		max_tokens: 100,
		metadata: None,
		system: None,
		stop_sequences: vec![],
		stream: false,
		temperature: None,
		top_k: None,
		top_p: None,
		tools: None,
		tool_choice: None,
		thinking: None,
		output_config: None,
	};

	let err = super::from_messages::translate_internal(req, &provider, None, None).unwrap_err();
	assert!(matches!(err, crate::AIError::UnsupportedConversion(_)));
	assert!(
		err
			.to_string()
			.contains("URL image sources are unsupported")
	);
}

#[test]
fn test_completions_image_data_url_maps_to_converse_image_block() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::completions::Request = serde_json::from_value(json!({
		"model": "gpt-4o",
		"max_tokens": 64,
		"messages": [{
			"role": "user",
			"content": [
				{ "type": "text", "text": "What is in this image?" },
				{
					"type": "image_url",
					"image_url": {
						"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
					}
				}
			]
		}]
	}))
	.expect("valid completions request");

	let translated = super::from_completions::translate(&req, &provider, None, None, None)
		.unwrap()
		.body;
	let translated: serde_json::Value = serde_json::from_slice(&translated).unwrap();

	let content = translated["messages"][0]["content"]
		.as_array()
		.expect("user message content");
	assert_eq!(content[0]["text"], json!("What is in this image?"));
	assert_eq!(content[1]["image"]["format"], json!("jpeg"));
	assert_eq!(
		content[1]["image"]["source"]["bytes"],
		json!("/9j/4AAQSkZJRg==")
	);
}

#[test]
fn test_completions_image_url_to_bedrock_returns_error() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};
	let req: types::completions::Request = serde_json::from_value(json!({
		"model": "gpt-4o",
		"messages": [{
			"role": "user",
			"content": [{
				"type": "image_url",
				"image_url": { "url": "https://example.com/sample.jpg" }
			}]
		}]
	}))
	.expect("valid completions request");

	let err = super::from_completions::translate(&req, &provider, None, None, None).unwrap_err();
	assert!(matches!(err, crate::AIError::UnsupportedConversion(_)));
}

#[test]
fn test_completions_request_metadata_only_uses_bedrock_header() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = types::completions::typed::Request {
		model: Some("anthropic.claude-3-sonnet".to_string()),
		moderation: None,
		messages: vec![types::completions::typed::RequestMessage::User(
			types::completions::typed::RequestUserMessage {
				content: types::completions::typed::RequestUserMessageContent::Text("Hello".to_string()),
				name: None,
			},
		)],
		stream: None,
		temperature: None,
		top_p: None,
		max_completion_tokens: Some(16),
		stop: None,
		tools: None,
		tool_choice: None,
		parallel_tool_calls: None,
		user: Some("user456".to_string()),
		vendor_extensions: Default::default(),
		frequency_penalty: None,
		logit_bias: None,
		logprobs: None,
		top_logprobs: None,
		n: None,
		modalities: None,
		prediction: None,
		audio: None,
		presence_penalty: None,
		response_format: None,
		seed: None,
		#[allow(deprecated)]
		function_call: None,
		#[allow(deprecated)]
		functions: None,
		metadata: Some(json!({
			"user_id": "user123",
			"department": "engineering",
			"json_user": r#"{"device_id":"from-body"}"#,
			"nonstr": 123
		})),
		#[allow(deprecated)]
		max_tokens: None,
		service_tier: None,
		web_search_options: None,
		stream_options: None,
		store: None,
		reasoning_effort: None,
	};
	let mut headers = HeaderMap::new();
	headers.insert(
		"x-bedrock-metadata",
		r#"{"json_user": "{\"device_id\":\"from-header\"}", "bad?key": "bad{}"}"#
			.parse()
			.unwrap(),
	);

	let (out, _) = super::from_completions::translate_internal(
		req,
		"anthropic.claude-3-sonnet".to_string(),
		&provider,
		Some(&headers),
		None,
		None,
	)
	.unwrap();
	let md = out.request_metadata.unwrap();

	assert!(!md.contains_key("user_id"));
	assert!(!md.contains_key("department"));
	assert_eq!(
		md.get("json_user"),
		Some(&r#"{"device_id":"from-header"}"#.to_string())
	);
	assert_eq!(md.get("bad?key"), Some(&"bad{}".to_string()));
	assert!(!md.contains_key("nonstr"));
}

#[test]
fn test_completions_json_schema_response_format_maps_to_converse_output_config() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let schema = json!({
		"type": "object",
		"properties": {
			"summary": { "type": "string" }
		},
		"required": ["summary"],
		"additionalProperties": false
	});

	let req = types::completions::typed::Request {
		model: Some("anthropic.claude-3-sonnet".to_string()),
		moderation: None,
		messages: vec![types::completions::typed::RequestMessage::User(
			types::completions::typed::RequestUserMessage {
				content: types::completions::typed::RequestUserMessageContent::Text(
					"Summarize".to_string(),
				),
				name: None,
			},
		)],
		stream: None,
		temperature: None,
		top_p: None,
		max_completion_tokens: Some(16),
		stop: None,
		tools: None,
		tool_choice: None,
		parallel_tool_calls: None,
		user: None,
		vendor_extensions: Default::default(),
		frequency_penalty: None,
		logit_bias: None,
		logprobs: None,
		top_logprobs: None,
		n: None,
		modalities: None,
		prediction: None,
		audio: None,
		presence_penalty: None,
		response_format: Some(types::completions::typed::ResponseFormat::JsonSchema {
			json_schema: types::completions::typed::ResponseFormatJsonSchema {
				description: Some("Structured summary".to_string()),
				name: "summary_schema".to_string(),
				schema: schema.clone(),
				strict: Some(true),
			},
		}),
		seed: None,
		#[allow(deprecated)]
		function_call: None,
		#[allow(deprecated)]
		functions: None,
		metadata: None,
		#[allow(deprecated)]
		max_tokens: None,
		service_tier: None,
		web_search_options: None,
		stream_options: None,
		store: None,
		reasoning_effort: None,
	};

	let (out, _) = super::from_completions::translate_internal(
		req,
		"anthropic.claude-3-sonnet".to_string(),
		&provider,
		None,
		None,
		None,
	)
	.unwrap();
	assert_eq!(
		out.output_config,
		Some(types::bedrock::OutputConfig {
			text_format: Some(types::bedrock::OutputFormat {
				r#type: types::bedrock::OutputFormatType::JsonSchema,
				structure: types::bedrock::OutputFormatStructure {
					json_schema: types::bedrock::JsonSchemaDefinition {
						schema: serde_json::to_string(&schema).unwrap(),
						name: Some("summary_schema".to_string()),
						description: Some("Structured summary".to_string()),
					},
				},
			}),
		})
	);
}

#[test]
fn test_completions_reasoning_effort_maps_to_enabled_thinking_budget() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = types::completions::typed::Request {
		model: Some("anthropic.claude-3-sonnet".to_string()),
		moderation: None,
		messages: vec![types::completions::typed::RequestMessage::User(
			types::completions::typed::RequestUserMessage {
				content: types::completions::typed::RequestUserMessageContent::Text(
					"Deeply analyze this topic".to_string(),
				),
				name: None,
			},
		)],
		stream: None,
		temperature: None,
		top_p: None,
		max_completion_tokens: Some(64),
		stop: None,
		tools: None,
		tool_choice: None,
		parallel_tool_calls: None,
		user: None,
		vendor_extensions: Default::default(),
		frequency_penalty: None,
		logit_bias: None,
		logprobs: None,
		top_logprobs: None,
		n: None,
		modalities: None,
		prediction: None,
		audio: None,
		presence_penalty: None,
		response_format: None,
		seed: None,
		#[allow(deprecated)]
		function_call: None,
		#[allow(deprecated)]
		functions: None,
		metadata: None,
		#[allow(deprecated)]
		max_tokens: None,
		service_tier: None,
		web_search_options: None,
		stream_options: None,
		store: None,
		reasoning_effort: Some(types::completions::typed::ReasoningEffort::Xhigh),
	};

	let (out, _) = super::from_completions::translate_internal(
		req,
		"anthropic.claude-3-sonnet".to_string(),
		&provider,
		None,
		None,
		None,
	)
	.unwrap();

	assert_eq!(
		out.additional_model_request_fields,
		Some(json!({
			"thinking": {
				"type": "enabled",
				"budget_tokens": 8192
			}
		}))
	);
}

#[test]
fn test_completions_explicit_thinking_budget_forces_enabled_thinking() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = types::completions::typed::Request {
		model: Some("anthropic.claude-3-sonnet".to_string()),
		moderation: None,
		messages: vec![types::completions::typed::RequestMessage::User(
			types::completions::typed::RequestUserMessage {
				content: types::completions::typed::RequestUserMessageContent::Text(
					"Deeply analyze this topic".to_string(),
				),
				name: None,
			},
		)],
		stream: None,
		temperature: None,
		top_p: None,
		max_completion_tokens: Some(64),
		stop: None,
		tools: None,
		tool_choice: None,
		parallel_tool_calls: None,
		user: None,
		vendor_extensions: types::completions::typed::RequestVendorExtensions {
			top_k: None,
			thinking_budget_tokens: Some(3072),
		},
		frequency_penalty: None,
		logit_bias: None,
		logprobs: None,
		top_logprobs: None,
		n: None,
		modalities: None,
		prediction: None,
		audio: None,
		presence_penalty: None,
		response_format: None,
		seed: None,
		#[allow(deprecated)]
		function_call: None,
		#[allow(deprecated)]
		functions: None,
		metadata: None,
		#[allow(deprecated)]
		max_tokens: None,
		service_tier: None,
		web_search_options: None,
		stream_options: None,
		store: None,
		reasoning_effort: Some(types::completions::typed::ReasoningEffort::High),
	};

	let (out, _) = super::from_completions::translate_internal(
		req,
		"anthropic.claude-3-sonnet".to_string(),
		&provider,
		None,
		None,
		None,
	)
	.unwrap();

	assert_eq!(
		out.additional_model_request_fields,
		Some(json!({
			"thinking": {
				"type": "enabled",
				"budget_tokens": 3072
			}
		}))
	);
}

#[test]
fn test_responses_request_metadata_only_uses_bedrock_header() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-4o",
		"max_output_tokens": 16,
		"input": "Hello",
		"metadata": {
			"safe": "ok",
			"json_user": "{\"device_id\":\"from-body\"}"
		}
	}))
	.expect("valid responses request");

	let mut headers = HeaderMap::new();
	headers.insert(
		"x-bedrock-metadata",
		r#"{"json_user": "{\"device_id\":\"from-header\"}", "bad?key": "bad{}"}"#
			.parse()
			.unwrap(),
	);

	let translated = super::from_responses::translate(&req, &provider, Some(&headers), None, None)
		.unwrap()
		.body;
	let translated: serde_json::Value = serde_json::from_slice(&translated).unwrap();
	let metadata = translated["requestMetadata"]
		.as_object()
		.expect("requestMetadata object");

	assert!(!metadata.contains_key("safe"));
	assert_eq!(
		translated["requestMetadata"]["json_user"],
		r#"{"device_id":"from-header"}"#
	);
	assert_eq!(translated["requestMetadata"]["bad?key"], "bad{}");
}

#[test]
fn test_responses_reasoning_effort_maps_to_enabled_thinking_budget() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-5",
		"max_output_tokens": 64,
		"input": "Classify the intent.",
		"reasoning": {
			"effort": "high"
		}
	}))
	.expect("valid responses request");

	let translated = super::from_responses::translate(&req, &provider, None, None, None)
		.unwrap()
		.body;
	let translated: serde_json::Value = serde_json::from_slice(&translated).unwrap();

	assert_eq!(
		translated["additionalModelRequestFields"],
		json!({
			"thinking": {
				"type": "enabled",
				"budget_tokens": 4096
			}
		})
	);
}

#[test]
fn test_responses_explicit_thinking_budget_forces_enabled_thinking() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-5",
		"max_output_tokens": 64,
		"input": "Classify the intent.",
		"reasoning": {
			"effort": "high"
		},
		"vendor_extensions": {
			"thinking_budget_tokens": 3072
		}
	}))
	.expect("valid responses request");

	let translated = super::from_responses::translate(&req, &provider, None, None, None)
		.unwrap()
		.body;
	let translated: serde_json::Value = serde_json::from_slice(&translated).unwrap();

	assert_eq!(
		translated["additionalModelRequestFields"],
		json!({
			"thinking": {
				"type": "enabled",
				"budget_tokens": 3072
			}
		})
	);
}

#[test]
fn test_responses_vendor_extension_thinking_budget_forces_enabled_thinking() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-5",
		"max_output_tokens": 64,
		"input": "Classify the intent.",
		"vendor_extensions": {
			"thinking_budget_tokens": 3072
		}
	}))
	.expect("valid responses request");

	let translated = super::from_responses::translate(&req, &provider, None, None, None)
		.unwrap()
		.body;
	let translated: serde_json::Value = serde_json::from_slice(&translated).unwrap();

	assert_eq!(
		translated["additionalModelRequestFields"],
		json!({
			"thinking": {
				"type": "enabled",
				"budget_tokens": 3072
			}
		})
	);
}

#[test]
fn test_embeddings_translation_titan() {
	let provider = Provider {
		model: Some(strng::new("amazon.titan-embed-text-v2:0")),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = types::embeddings::Request {
		model: Some("text-embedding-3-small".to_string()),
		input: json!("hello world"),
		user: None,
		encoding_format: None,
		dimensions: Some(1024),
		rest: json!({}),
	};

	let translated = from_embeddings::translate(&req, &provider).unwrap();
	let bedrock_req: bedrock::AmazonTitanV2EmbeddingRequest =
		serde_json::from_slice(&translated).unwrap();

	assert_eq!(bedrock_req.input_text, "hello world");
	assert_eq!(bedrock_req.dimensions, Some(1024));
}

#[test]
fn test_embeddings_titan_with_encoding_format() {
	let provider = Provider {
		model: Some(strng::new("amazon.titan-embed-text-v2:0")),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = types::embeddings::Request {
		model: Some("text-embedding-3-small".to_string()),
		input: json!("hello"),
		user: None,
		encoding_format: Some(types::embeddings::typed::EncodingFormat::Float),
		dimensions: None,
		rest: json!({"normalize": true}),
	};

	let translated = from_embeddings::translate(&req, &provider).unwrap();
	let bedrock_req: bedrock::AmazonTitanV2EmbeddingRequest =
		serde_json::from_slice(&translated).unwrap();

	assert_eq!(bedrock_req.normalize, Some(true));
	assert!(
		matches!(&bedrock_req.embedding_types, Some(v) if v.len() == 1),
		"expected one embedding type"
	);
}

#[test]
fn test_embeddings_titan_rejects_array_input() {
	let provider = Provider {
		model: Some(strng::new("amazon.titan-embed-text-v2:0")),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = types::embeddings::Request {
		model: Some("text-embedding-3-small".to_string()),
		input: json!(["hello", "world"]),
		user: None,
		encoding_format: None,
		dimensions: None,
		rest: json!({}),
	};

	assert!(
		from_embeddings::translate(&req, &provider).is_err(),
		"Titan should reject array input"
	);
}

#[test]
fn test_embeddings_cohere_with_passthrough_fields() {
	let provider = Provider {
		model: Some(strng::new("cohere.embed-english-v3")),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = types::embeddings::Request {
		model: Some("text-embedding-3-small".to_string()),
		input: json!(["hello", "world"]),
		user: None,
		encoding_format: None,
		dimensions: None,
		rest: json!({"input_type": "search_document", "truncate": "END"}),
	};

	let translated = from_embeddings::translate(&req, &provider).unwrap();
	let bedrock_req: bedrock::CohereEmbeddingRequest = serde_json::from_slice(&translated).unwrap();

	assert_eq!(bedrock_req.texts, vec!["hello", "world"]);
	assert_eq!(bedrock_req.input_type, "search_document");
	assert_eq!(bedrock_req.truncate, Some("END".to_string()));
	assert_eq!(bedrock_req.output_dimension, None);
}

#[test]
fn test_embeddings_translation_nova() {
	let provider = Provider {
		model: Some(strng::new("amazon.nova-2-multimodal-embeddings-v1:0")),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = types::embeddings::Request {
		model: Some("text-embedding-3-small".to_string()),
		input: json!("hello world"),
		user: None,
		encoding_format: None,
		dimensions: Some(1024),
		rest: json!({}),
	};

	let translated = from_embeddings::translate(&req, &provider).unwrap();
	let bedrock_req: serde_json::Value = serde_json::from_slice(&translated).unwrap();

	assert_eq!(bedrock_req["taskType"], "SINGLE_EMBEDDING");
	let params = &bedrock_req["singleEmbeddingParams"];
	assert_eq!(params["embeddingPurpose"], "GENERIC_INDEX");
	assert_eq!(params["embeddingDimension"], 1024);
	assert_eq!(params["text"]["truncationMode"], "END");
	assert_eq!(params["text"]["value"], "hello world");
}

#[test]
fn test_embeddings_nova_omits_dimension_when_unset() {
	let provider = Provider {
		model: Some(strng::new("amazon.nova-2-multimodal-embeddings-v1:0")),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = types::embeddings::Request {
		model: Some("text-embedding-3-small".to_string()),
		input: json!("hello"),
		user: None,
		encoding_format: None,
		dimensions: None,
		rest: json!({}),
	};

	let translated = from_embeddings::translate(&req, &provider).unwrap();
	let bedrock_req: serde_json::Value = serde_json::from_slice(&translated).unwrap();

	assert!(
		bedrock_req["singleEmbeddingParams"]
			.get("embeddingDimension")
			.is_none(),
		"embeddingDimension should be omitted so the model default applies"
	);
}

#[test]
fn test_embeddings_nova_with_passthrough_fields() {
	let provider = Provider {
		model: Some(strng::new("amazon.nova-2-multimodal-embeddings-v1:0")),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = types::embeddings::Request {
		model: Some("text-embedding-3-small".to_string()),
		input: json!("hello"),
		user: None,
		encoding_format: None,
		dimensions: None,
		rest: json!({"embedding_purpose": "GENERIC_RETRIEVAL", "truncation_mode": "NONE"}),
	};

	let translated = from_embeddings::translate(&req, &provider).unwrap();
	let bedrock_req: bedrock::NovaEmbeddingRequest = serde_json::from_slice(&translated).unwrap();

	assert_eq!(
		bedrock_req.single_embedding_params.embedding_purpose,
		"GENERIC_RETRIEVAL"
	);
	assert_eq!(
		bedrock_req.single_embedding_params.text.truncation_mode,
		"NONE"
	);
}

#[test]
fn test_embeddings_nova_rejects_array_input() {
	let provider = Provider {
		model: Some(strng::new("amazon.nova-2-multimodal-embeddings-v1:0")),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = types::embeddings::Request {
		model: Some("text-embedding-3-small".to_string()),
		input: json!(["hello", "world"]),
		user: None,
		encoding_format: None,
		dimensions: None,
		rest: json!({}),
	};

	assert!(
		from_embeddings::translate(&req, &provider).is_err(),
		"Nova should reject array input"
	);
}

#[test]
fn test_embeddings_rejects_invalid_input() {
	let provider = Provider {
		model: Some(strng::new("cohere.embed-english-v3")),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	for input in [json!(["hello", 42]), json!(42)] {
		let req = types::embeddings::Request {
			model: Some("text-embedding-3-small".to_string()),
			input,
			user: None,
			encoding_format: None,
			dimensions: None,
			rest: json!({}),
		};
		assert!(from_embeddings::translate(&req, &provider).is_err());
	}
}

#[test]
fn test_embeddings_response_translation_titan() {
	let model = "amazon.titan-embed-text-v2:0";
	let bedrock_resp = json!({
		"embedding": [0.1, 0.2, 0.3],
		"inputTextTokenCount": 3
	});
	let bytes = serde_json::to_vec(&bedrock_resp).unwrap();
	let headers = HeaderMap::new();

	let translated = from_embeddings::translate_response(&bytes, &headers, model).unwrap();
	let openai_resp = translated
		.serialize()
		.and_then(|b| serde_json::from_slice::<types::embeddings::Response>(&b))
		.unwrap();

	assert_eq!(openai_resp.object, "list");
	assert_eq!(openai_resp.usage.unwrap().prompt_tokens, 3);
}

#[test]
fn test_embeddings_response_titan_embeddings_by_type_fallback() {
	let model = "amazon.titan-embed-text-v2:0";
	let bedrock_resp = json!({
		"embeddingsByType": {
			"float": [0.4, 0.5, 0.6]
		},
		"inputTextTokenCount": 5
	});
	let bytes = serde_json::to_vec(&bedrock_resp).unwrap();
	let headers = HeaderMap::new();

	let translated = from_embeddings::translate_response(&bytes, &headers, model).unwrap();
	let openai_resp = translated
		.serialize()
		.and_then(|b| serde_json::from_slice::<types::embeddings::Response>(&b))
		.unwrap();

	assert_eq!(openai_resp.usage.unwrap().prompt_tokens, 5);
}

#[test]
fn test_embeddings_response_translation_cohere() {
	let model = "cohere.embed-english-v3";
	let bedrock_resp = json!({
		"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
		"id": "123",
		"texts": ["hello", "world"]
	});
	let bytes = serde_json::to_vec(&bedrock_resp).unwrap();
	let mut headers = HeaderMap::new();
	headers.insert("x-amzn-bedrock-input-token-count", "10".parse().unwrap());

	let translated = from_embeddings::translate_response(&bytes, &headers, model).unwrap();
	let openai_resp = translated
		.serialize()
		.and_then(|b| serde_json::from_slice::<types::embeddings::typed::Response>(&b))
		.unwrap();

	assert_eq!(openai_resp.object, "list");
	assert_eq!(
		openai_resp.data[0].embedding,
		vec![0.1_f32, 0.2_f32, 0.3_f32]
	);
	assert_eq!(
		openai_resp.data[1].embedding,
		vec![0.4_f32, 0.5_f32, 0.6_f32]
	);
	assert_eq!(openai_resp.usage.prompt_tokens, 10);
}

#[test]
fn test_embeddings_response_translation_cohere_v4_uses_float_vectors() {
	let model = "cohere.embed-v4:0";
	let bedrock_resp = json!({
		"embeddings": {
			"float": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
			"int8": [[1, 2, 3], [4, 5, 6]]
		},
		"id": "123",
		"texts": ["hello", "world"]
	});
	let bytes = serde_json::to_vec(&bedrock_resp).unwrap();
	let headers = HeaderMap::new();

	let translated = from_embeddings::translate_response(&bytes, &headers, model).unwrap();
	let openai_resp = translated
		.serialize()
		.and_then(|b| serde_json::from_slice::<types::embeddings::typed::Response>(&b))
		.unwrap();

	assert_eq!(
		openai_resp.data[0].embedding,
		vec![0.1_f32, 0.2_f32, 0.3_f32]
	);
	assert_eq!(
		openai_resp.data[1].embedding,
		vec![0.4_f32, 0.5_f32, 0.6_f32]
	);
}

#[test]
fn test_embeddings_response_translation_cohere_v4_requires_float_vectors() {
	let model = "cohere.embed-v4:0";
	let bedrock_resp = json!({
		"embeddings": {
			"uint8": [[1, 2, 3]],
			"int8": [[-1, 0, 1]]
		},
		"id": "123",
		"texts": ["hello"]
	});
	let bytes = serde_json::to_vec(&bedrock_resp).unwrap();
	let headers = HeaderMap::new();

	let err = match from_embeddings::translate_response(&bytes, &headers, model) {
		Ok(_) => panic!("expected a response without float embeddings to fail"),
		Err(err) => err,
	};

	assert!(matches!(err, crate::AIError::ResponseParsing(_)));
	assert!(
		err
			.to_string()
			.contains("Cohere response did not include float embeddings; received types: int8, uint8")
	);
}

#[test]
fn test_embeddings_response_translation_nova() {
	let model = "amazon.nova-2-multimodal-embeddings-v1:0";
	let bedrock_resp = json!({
		"embeddings": [{"embedding": [0.25, 0.5, -0.75], "embeddingType": "TEXT"}]
	});
	let bytes = serde_json::to_vec(&bedrock_resp).unwrap();
	let mut headers = HeaderMap::new();
	headers.insert("x-amzn-bedrock-input-token-count", "7".parse().unwrap());

	let translated = from_embeddings::translate_response(&bytes, &headers, model).unwrap();
	let openai_resp: serde_json::Value = translated
		.serialize()
		.and_then(|b| serde_json::from_slice(&b))
		.unwrap();

	assert_eq!(openai_resp["object"], "list");
	assert_eq!(openai_resp["data"][0]["object"], "embedding");
	assert_eq!(openai_resp["data"][0]["index"], 0);
	assert_eq!(
		openai_resp["data"][0]["embedding"],
		json!([0.25, 0.5, -0.75])
	);
	assert_eq!(openai_resp["usage"]["prompt_tokens"], 7);
	assert_eq!(openai_resp["usage"]["total_tokens"], 7);
}

#[test]
fn test_embeddings_error_translation() {
	let error_body =
		bytes::Bytes::from(serde_json::to_vec(&json!({"message": "Model not found"})).unwrap());

	let translated = from_embeddings::translate_error(&error_body).unwrap();
	let error_resp: serde_json::Value = serde_json::from_slice(&translated).unwrap();

	assert_eq!(error_resp["error"]["type"], "invalid_request_error");
	assert_eq!(error_resp["error"]["message"], "Model not found");
}

#[test]
fn test_completions_error_translation_wraps_non_json_body() {
	let error_body = Bytes::from_static(
		b"<html><body><center>The plain HTTP request was sent to HTTPS port</center></body></html>",
	);

	let translated = from_completions::translate_error(&error_body).unwrap();
	let error_resp: serde_json::Value = serde_json::from_slice(&translated).unwrap();

	assert_eq!(error_resp["error"]["type"], "invalid_request_error");
	assert!(
		error_resp["error"]["message"]
			.as_str()
			.unwrap()
			.contains("plain HTTP request")
	);
}

fn make_message(role: types::bedrock::Role, text: &str) -> types::bedrock::Message {
	types::bedrock::Message {
		role,
		content: vec![types::bedrock::ContentBlock::Text(text.to_string())],
	}
}

fn has_cache_point(msg: &types::bedrock::Message) -> bool {
	msg
		.content
		.iter()
		.any(|b| matches!(b, types::bedrock::ContentBlock::CachePoint(_)))
}

#[test]
fn test_insert_cache_point_default_offset() {
	let mut msgs = vec![
		make_message(types::bedrock::Role::User, "Hello"),
		make_message(types::bedrock::Role::Assistant, "Hi"),
		make_message(types::bedrock::Role::User, "How are you?"),
	];
	helpers::insert_message_cache_point(&mut msgs, 0);
	assert!(has_cache_point(&msgs[1]));
	assert!(!has_cache_point(&msgs[0]));
	assert!(!has_cache_point(&msgs[2]));
}

#[test]
fn test_insert_cache_point_offset_shifts_back() {
	let mut msgs = vec![
		make_message(types::bedrock::Role::User, "a"),
		make_message(types::bedrock::Role::Assistant, "b"),
		make_message(types::bedrock::Role::User, "c"),
		make_message(types::bedrock::Role::Assistant, "d"),
		make_message(types::bedrock::Role::User, "e"),
	];
	helpers::insert_message_cache_point(&mut msgs, 2);
	// default position is index 3 (len-2), offset 2 → index 1
	assert!(has_cache_point(&msgs[1]));
	for (i, msg) in msgs.iter().enumerate() {
		if i != 1 {
			assert!(!has_cache_point(msg));
		}
	}
}

#[test]
fn test_insert_cache_point_offset_clamps_to_zero() {
	let mut msgs = vec![
		make_message(types::bedrock::Role::User, "a"),
		make_message(types::bedrock::Role::Assistant, "b"),
		make_message(types::bedrock::Role::User, "c"),
	];
	// offset 100 should clamp to index 0
	helpers::insert_message_cache_point(&mut msgs, 100);
	assert!(has_cache_point(&msgs[0]));
	assert!(!has_cache_point(&msgs[1]));
	assert!(!has_cache_point(&msgs[2]));
}

#[test]
fn test_insert_cache_point_single_message_noop() {
	let mut msgs = vec![make_message(types::bedrock::Role::User, "only")];
	helpers::insert_message_cache_point(&mut msgs, 0);
	assert!(!has_cache_point(&msgs[0]));
}

#[test]
fn test_insert_cache_point_empty_messages_noop() {
	let mut msgs: Vec<types::bedrock::Message> = vec![];
	helpers::insert_message_cache_point(&mut msgs, 0);
	assert!(msgs.is_empty());
}

#[test]
fn test_bedrock_tool_name_sanitizes_long_mcp_names() {
	let long_name = "mcp__plugin_atlassian_atlassian__createCompassComponentRelationship";
	assert!(long_name.len() > super::BEDROCK_TOOL_NAME_MAX_LEN);

	let mut map = super::BedrockToolNameMap::default();
	let sanitized = map.register(long_name);

	assert!(sanitized.len() <= super::BEDROCK_TOOL_NAME_MAX_LEN);
	assert!(
		sanitized
			.chars()
			.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
	);
	assert_eq!(map.restore(&sanitized), long_name);
}

#[test]
fn test_bedrock_tool_name_preserves_valid_short_names() {
	let mut map = super::BedrockToolNameMap::default();
	let name = "get_weather";
	assert_eq!(map.register(name), name);
	assert_eq!(map.restore(name), name);
	assert!(map.is_empty());
}

#[test]
fn test_bedrock_tool_name_sanitizes_invalid_characters() {
	let mut map = super::BedrockToolNameMap::default();
	let sanitized = map.register("my.tool/name");
	assert_eq!(sanitized, "my_tool_name");
}

#[test]
fn test_messages_long_tool_names_fit_bedrock_tool_config() {
	use types::messages::typed as messages;

	let long_name = "mcp__plugin_atlassian_atlassian__createCompassComponentRelationship";
	let provider = Provider {
		model: None,
		region: strng::new("us-west-2"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = messages::Request {
		model: "anthropic.claude-sonnet-4-20250514-v1:0".to_string(),
		max_tokens: 1024,
		messages: vec![messages::Message {
			role: messages::Role::User,
			content: vec![messages::ContentBlock::Text(messages::ContentTextBlock {
				text: "hello".to_string(),
				citations: None,
				cache_control: None,
			})],
		}],
		tools: Some(vec![messages::Tool::Custom(messages::CustomTool {
			name: long_name.to_string(),
			description: Some("test".to_string()),
			input_schema: serde_json::json!({"type": "object"}),
			cache_control: None,
		})]),
		tool_choice: None,
		system: None,
		metadata: None,
		stop_sequences: vec![],
		stream: false,
		temperature: None,
		top_p: None,
		top_k: None,
		thinking: None,
		output_config: None,
	};

	let (out, tool_map) =
		super::from_messages::translate_internal(req, &provider, None, None).unwrap();
	let bedrock_name = out
		.tool_config
		.as_ref()
		.and_then(|tc| tc.tools.first())
		.and_then(|tool| match tool {
			types::bedrock::Tool::ToolSpec(spec) => Some(spec.name.clone()),
			_ => None,
		})
		.expect("tool spec");

	assert!(bedrock_name.len() <= super::BEDROCK_TOOL_NAME_MAX_LEN);
	assert_eq!(tool_map.restore(&bedrock_name), long_name);
}

#[test]
fn test_messages_long_tool_name_round_trip_response() {
	use types::messages::typed as messages;

	let long_name = "mcp__plugin_atlassian_atlassian__createCompassComponentRelationship";
	let provider = Provider {
		model: None,
		region: strng::new("us-west-2"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req = messages::Request {
		model: "anthropic.claude-sonnet-4-20250514-v1:0".to_string(),
		max_tokens: 1024,
		messages: vec![messages::Message {
			role: messages::Role::User,
			content: vec![messages::ContentBlock::Text(messages::ContentTextBlock {
				text: "hello".to_string(),
				citations: None,
				cache_control: None,
			})],
		}],
		tools: Some(vec![messages::Tool::Custom(messages::CustomTool {
			name: long_name.to_string(),
			description: Some("test".to_string()),
			input_schema: serde_json::json!({"type": "object"}),
			cache_control: None,
		})]),
		tool_choice: None,
		system: None,
		metadata: None,
		stop_sequences: vec![],
		stream: false,
		temperature: None,
		top_p: None,
		top_k: None,
		thinking: None,
		output_config: None,
	};

	let (bedrock_req, tool_map) =
		super::from_messages::translate_internal(req, &provider, None, None).unwrap();
	let bedrock_name = bedrock_req
		.tool_config
		.as_ref()
		.and_then(|tc| tc.tools.first())
		.and_then(|tool| match tool {
			types::bedrock::Tool::ToolSpec(spec) => Some(spec.name.clone()),
			_ => None,
		})
		.expect("sanitized tool name");
	let model = "anthropic.claude-sonnet-4-20250514-v1:0";

	let bedrock_response = json!({
		"output": {
			"message": {
				"role": "assistant",
				"content": [{
					"toolUse": {
						"toolUseId": "toolu_01TestRoundTrip",
						"name": bedrock_name,
						"input": {"query": "test"}
					}
				}]
			}
		},
		"stopReason": "tool_use",
		"usage": {
			"inputTokens": 100,
			"outputTokens": 20,
			"totalTokens": 120
		}
	});
	let bytes = Bytes::from(serde_json::to_vec(&bedrock_response).unwrap());

	let response = super::from_messages::translate_response(&bytes, model, Some(&tool_map)).unwrap();
	let response_json: serde_json::Value =
		serde_json::from_slice(&response.serialize().unwrap()).unwrap();
	let tool_use_name = response_json["content"]
		.as_array()
		.and_then(|blocks| {
			blocks.iter().find_map(|block| {
				block
					.get("name")
					.and_then(|n| n.as_str())
					.filter(|_| block.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
			})
		})
		.expect("tool use block in response");

	assert_eq!(tool_use_name, long_name);
}

#[test]
fn test_responses_assistant_input_image_is_rejected() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-4o",
		"max_output_tokens": 64,
		"input": [{
			"role": "assistant",
			"content": [{
				"type": "input_image",
				"image_url": "data:image/png;base64,iVBORw0KGgo=",
				"detail": "auto"
			}]
		}]
	}))
	.expect("valid responses request");

	let err = super::from_responses::translate(&req, &provider, None, None, None).unwrap_err();
	assert!(matches!(err, crate::AIError::UnsupportedConversion(_)));
	assert!(
		err
			.to_string()
			.contains("image inputs are only supported on user messages")
	);
}

#[test]
fn test_responses_input_image_remote_url_is_rejected() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-4o",
		"max_output_tokens": 64,
		"input": [{
			"role": "user",
			"content": [{
				"type": "input_image",
				"image_url": "https://example.com/sample.png",
				"detail": "auto"
			}]
		}]
	}))
	.expect("valid responses request");

	let err = super::from_responses::translate(&req, &provider, None, None, None).unwrap_err();
	assert!(matches!(err, crate::AIError::UnsupportedConversion(_)));
	assert!(
		err
			.to_string()
			.contains("remote URLs and file_ids are unsupported")
	);
}

#[test]
fn test_responses_input_image_non_base64_data_url_is_rejected() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-4o",
		"max_output_tokens": 64,
		"input": [{
			"role": "user",
			"content": [{
				"type": "input_image",
				"image_url": "data:image/png,iVBORw0KGgo=",
				"detail": "auto"
			}]
		}]
	}))
	.expect("valid responses request");

	let err = super::from_responses::translate(&req, &provider, None, None, None).unwrap_err();
	assert!(matches!(err, crate::AIError::UnsupportedConversion(_)));
	assert!(
		err
			.to_string()
			.contains("image data URLs must be base64-encoded")
	);
}

#[test]
fn test_responses_input_image_non_image_data_url_is_rejected() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-4o",
		"max_output_tokens": 64,
		"input": [{
			"role": "user",
			"content": [{
				"type": "input_image",
				"image_url": "data:application/octet-stream;base64,iVBORw0KGgo=",
				"detail": "auto"
			}]
		}]
	}))
	.expect("valid responses request");

	let err = super::from_responses::translate(&req, &provider, None, None, None).unwrap_err();
	assert!(matches!(err, crate::AIError::UnsupportedConversion(_)));
	assert!(
		err
			.to_string()
			.contains("image data URLs must use a non-empty image/* media type")
	);
}

#[test]
fn test_responses_input_image_empty_media_type_data_url_is_rejected() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-4o",
		"max_output_tokens": 64,
		"input": [{
			"role": "user",
			"content": [{
				"type": "input_image",
				"image_url": "data:;base64,iVBORw0KGgo=",
				"detail": "auto"
			}]
		}]
	}))
	.expect("valid responses request");

	let err = super::from_responses::translate(&req, &provider, None, None, None).unwrap_err();
	assert!(matches!(err, crate::AIError::UnsupportedConversion(_)));
	assert!(
		err
			.to_string()
			.contains("image data URLs must use a non-empty image/* media type")
	);
}

#[test]
fn test_responses_input_image_file_id_is_rejected() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-4o",
		"max_output_tokens": 64,
		"input": [{
			"role": "user",
			"content": [{
				"type": "input_image",
				"file_id": "file-abc123",
				"detail": "auto"
			}]
		}]
	}))
	.expect("valid responses request");

	let err = super::from_responses::translate(&req, &provider, None, None, None).unwrap_err();
	assert!(matches!(err, crate::AIError::UnsupportedConversion(_)));
}

#[test]
fn test_responses_system_input_file_is_rejected() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-4o",
		"max_output_tokens": 64,
		"input": [{
			"type": "message",
			"role": "system",
			"content": [{
				"type": "input_file",
				"file_id": "file-abc123"
			}]
		}]
	}))
	.expect("valid responses request");

	let err = super::from_responses::translate(&req, &provider, None, None, None).unwrap_err();
	assert!(matches!(err, crate::AIError::UnsupportedConversion(_)));
	assert!(
		err
			.to_string()
			.contains("bedrock document inputs are only supported on user messages"),
		"unexpected error: {err}"
	);
}

#[test]
fn test_responses_input_file_id_is_rejected() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-4o",
		"max_output_tokens": 64,
		"input": [{
			"role": "user",
			"content": [{
				"type": "input_file",
				"file_id": "file-abc123"
			}]
		}]
	}))
	.expect("valid responses request");

	let err = super::from_responses::translate(&req, &provider, None, None, None).unwrap_err();
	assert!(matches!(err, crate::AIError::UnsupportedConversion(_)));
	assert!(
		err.to_string().contains("file_id is unsupported"),
		"unexpected error: {err}"
	);
}

#[test]
fn test_responses_input_file_remote_url_is_rejected() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-4o",
		"max_output_tokens": 64,
		"input": [{
			"role": "user",
			"content": [{
				"type": "input_file",
				"file_url": "https://example.com/report.pdf",
				"filename": "report.pdf"
			}]
		}]
	}))
	.expect("valid responses request");

	let err = super::from_responses::translate(&req, &provider, None, None, None).unwrap_err();
	assert!(matches!(err, crate::AIError::UnsupportedConversion(_)));
	assert!(
		err.to_string().contains("remote URLs are unsupported"),
		"unexpected error: {err}"
	);
}

#[test]
fn test_responses_input_file_unknown_format_is_rejected() {
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};

	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "gpt-4o",
		"max_output_tokens": 64,
		"input": [{
			"role": "user",
			"content": [{
				"type": "input_file",
				"file_data": "data:application/octet-stream;base64,dGVzdA==",
				"filename": "archive.zip"
			}]
		}]
	}))
	.expect("valid responses request");

	let err = super::from_responses::translate(&req, &provider, None, None, None).unwrap_err();
	assert!(matches!(err, crate::AIError::UnsupportedConversion(_)));
	assert!(
		err
			.to_string()
			.contains("document format could not be determined"),
		"unexpected error: {err}"
	);
}

#[test]
fn test_reasoning_content_block_deserializes_redacted_content() {
	// xAI Grok 4.6 on Bedrock Converse returns encrypted reasoning as
	// { "reasoningContent": { "redactedContent": "<base64>" } } in
	// non-streaming responses; this must parse, not 502.
	let block: types::bedrock::ContentBlock = serde_json::from_value(json!({
		"reasoningContent": { "redactedContent": "cnNuX2pRenlYWklGSkxscEt5ZmxZa0NmdndJRmVMVjVaYkRXWDRo" }
	}))
	.expect("redactedContent variant must deserialize");
	match block {
		types::bedrock::ContentBlock::ReasoningContent(
			types::bedrock::ReasoningContentBlock::Redacted { redacted_content },
		) => {
			assert_eq!(
				redacted_content,
				"cnNuX2pRenlYWklGSkxscEt5ZmxZa0NmdndJRmVMVjVaYkRXWDRo"
			);
		},
		other => panic!("expected Redacted variant, got {other:?}"),
	}

	// The pre-existing variants must still take priority.
	let structured: types::bedrock::ContentBlock = serde_json::from_value(json!({
		"reasoningContent": { "reasoningText": { "text": "thinking...", "signature": "sig" } }
	}))
	.expect("structured variant");
	assert!(matches!(
		structured,
		types::bedrock::ContentBlock::ReasoningContent(
			types::bedrock::ReasoningContentBlock::Structured { .. }
		)
	));
}

#[test]
fn test_redacted_thinking_round_trips_to_bedrock_request() {
	// A redacted_thinking block returned to the client must be forwarded back
	// to Bedrock as ReasoningContentBlock::Redacted on the next turn, so
	// multi-turn conversations can replay the opaque payload.
	let provider = Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};
	let payload = "cnNuX2pRenlYWklGSkxscEt5ZmxZa0NmdndJRmVMVjVaYkRXWDRo";
	let req = messages::typed::Request {
		model: "us.xai.grok-4.6".to_string(),
		messages: vec![
			messages::typed::Message {
				role: messages::typed::Role::User,
				content: vec![messages::typed::ContentBlock::Text(
					messages::typed::ContentTextBlock {
						text: "Say ok.".to_string(),
						citations: None,
						cache_control: None,
					},
				)],
			},
			messages::typed::Message {
				role: messages::typed::Role::Assistant,
				content: vec![
					messages::typed::ContentBlock::RedactedThinking {
						data: payload.to_string(),
					},
					messages::typed::ContentBlock::Text(messages::typed::ContentTextBlock {
						text: "ok".to_string(),
						citations: None,
						cache_control: None,
					}),
				],
			},
		],
		max_tokens: 100,
		metadata: None,
		system: None,
		stop_sequences: vec![],
		stream: false,
		temperature: None,
		top_k: None,
		top_p: None,
		tools: None,
		tool_choice: None,
		thinking: None,
		output_config: None,
	};

	let (out, _) = super::from_messages::translate_internal(req, &provider, None, None).unwrap();
	let assistant_content = &out.messages[1].content;
	match &assistant_content[0] {
		types::bedrock::ContentBlock::ReasoningContent(
			types::bedrock::ReasoningContentBlock::Redacted { redacted_content },
		) => assert_eq!(redacted_content, payload),
		other => panic!("expected Redacted reasoning block first, got {other:?}"),
	}
}

#[test]
fn test_tool_result_tool_reference_does_not_reject_the_request() {
	// Before ToolResultContentPart gained this variant, an untagged-enum mismatch
	// made the captured block a 400 with no upstream call. Converse still cannot
	// activate the tools -- only InvokeModel can -- but the request must survive and
	// carry the names.
	let provider = Provider {
		model: Some(strng::new("anthropic.claude-sonnet-4-6")),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};
	let req: types::messages::Request = serde_json::from_value(json!({
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages": [{
			"role": "user",
			"content": [{
				"type": "tool_result",
				"tool_use_id": "tooluse_36nMoA0hip1UR8bztp3jzT",
				"content": [
					{"type": "tool_reference", "tool_name": "mcp__example__list_schedules"},
					{"type": "tool_reference", "tool_name": "mcp__example__list_schedule_occurrences"}
				]
			}]
		}]
	}))
	.unwrap();

	let translated = super::from_messages::translate(&req, &provider, None, None)
		.expect("a tool_reference block must not fail the request");
	let body: serde_json::Value = serde_json::from_slice(&translated.body).unwrap();
	let content = &body["messages"][0]["content"][0]["toolResult"]["content"];
	assert_eq!(
		content,
		&json!([
			{"json": {"type": "tool_reference", "tool_name": "mcp__example__list_schedules"}},
			{"json": {"type": "tool_reference", "tool_name": "mcp__example__list_schedule_occurrences"}}
		]),
		"got {body}"
	);
}

#[test]
fn test_unknown_tool_result_part_is_dropped_not_rejected() {
	// A content type this build predates must cost the block, never the request.
	let provider = Provider {
		model: Some(strng::new("anthropic.claude-sonnet-4-6")),
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	};
	let req: types::messages::Request = serde_json::from_value(json!({
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages": [{
			"role": "user",
			"content": [{
				"type": "tool_result",
				"tool_use_id": "tooluse_1",
				"content": [
					{"type": "some_future_block_type", "whatever": 1},
					{"type": "text", "text": "kept"}
				]
			}]
		}]
	}))
	.unwrap();

	let translated = super::from_messages::translate(&req, &provider, None, None)
		.expect("an unknown tool_result part must not fail the request");
	let body: serde_json::Value = serde_json::from_slice(&translated.body).unwrap();
	assert_eq!(
		&body["messages"][0]["content"][0]["toolResult"]["content"],
		&json!([{"text": "kept"}]),
		"got {body}"
	);
}

fn reasoning_effort_request(model: &str, effort: &str) -> types::completions::Request {
	serde_json::from_value(json!({
		"model": model,
		"reasoning_effort": effort,
		"max_completion_tokens": 512,
		"messages": [{ "role": "user", "content": "hello" }]
	}))
	.expect("valid completions request")
}

fn responses_reasoning_effort_request(model: &str, effort: &str) -> types::responses::Request {
	serde_json::from_value(json!({
		"model": model,
		"reasoning": { "effort": effort },
		"max_output_tokens": 512,
		"input": "hello"
	}))
	.expect("valid responses request")
}

fn reasoning_test_provider() -> Provider {
	Provider {
		model: None,
		region: strng::new("us-east-1"),
		guardrail_identifier: None,
		guardrail_version: None,
	}
}

fn translated_additional_fields(model: &str, effort: &str) -> serde_json::Value {
	let body = super::from_completions::translate(
		&reasoning_effort_request(model, effort),
		&reasoning_test_provider(),
		None,
		None,
		None,
	)
	.unwrap()
	.body;
	let translated: serde_json::Value = serde_json::from_slice(&body).unwrap();
	translated["additionalModelRequestFields"].clone()
}

fn responses_translated_additional_fields(model: &str, effort: &str) -> serde_json::Value {
	let body = super::from_responses::translate(
		&responses_reasoning_effort_request(model, effort),
		&reasoning_test_provider(),
		None,
		None,
		None,
	)
	.unwrap()
	.body;
	let translated: serde_json::Value = serde_json::from_slice(&body).unwrap();
	translated["additionalModelRequestFields"].clone()
}

#[test]
fn test_completions_reasoning_effort_on_openai_bedrock_model_maps_to_reasoning_effort_field() {
	let fields = translated_additional_fields("global.openai.gpt-5.6-terra", "medium");
	assert_eq!(fields, json!({ "reasoning": { "effort": "medium" } }));
	assert!(fields.get("thinking").is_none());
}

#[test]
fn test_completions_reasoning_effort_minimal_on_openai_bedrock_model_maps_to_low() {
	let fields = translated_additional_fields("openai.gpt-oss-120b-1:0", "minimal");
	assert_eq!(fields, json!({ "reasoning": { "effort": "low" } }));
}

#[test]
fn test_completions_reasoning_effort_none_on_openai_bedrock_model_sends_no_reasoning_fields() {
	let fields = translated_additional_fields("us.openai.gpt-5.6-sol", "none");
	assert!(fields.is_null(), "got {fields}");
}

#[test]
fn test_completions_reasoning_effort_on_claude_bedrock_model_still_maps_to_thinking_budget() {
	let fields = translated_additional_fields("global.anthropic.claude-sonnet-5", "medium");
	assert_eq!(
		fields,
		json!({ "thinking": { "type": "enabled", "budget_tokens": 2048 } })
	);
	assert!(fields.get("reasoning").is_none());
}

#[test]
fn test_responses_reasoning_effort_on_openai_bedrock_model_maps_to_reasoning_effort_field() {
	// Codex CLI sends `reasoning: {effort}` on every turn against gpt-5.6; the gateway used to
	// translate it into Anthropic's `thinking` object, which Bedrock rejects for OpenAI models.
	let fields = responses_translated_additional_fields("global.openai.gpt-5.6-sol", "high");
	assert_eq!(fields, json!({ "reasoning": { "effort": "high" } }));
	assert!(fields.get("thinking").is_none());
}

#[test]
fn test_responses_reasoning_effort_minimal_on_openai_bedrock_model_maps_to_low() {
	let fields = responses_translated_additional_fields("openai.gpt-oss-120b-1:0", "minimal");
	assert_eq!(fields, json!({ "reasoning": { "effort": "low" } }));
}

#[test]
fn test_responses_reasoning_effort_none_on_openai_bedrock_model_sends_no_reasoning_fields() {
	let fields = responses_translated_additional_fields("us.openai.gpt-5.6-sol", "none");
	assert!(fields.is_null(), "got {fields}");
}

#[test]
fn test_responses_reasoning_effort_on_claude_bedrock_model_still_maps_to_thinking() {
	let fields = responses_translated_additional_fields("global.anthropic.claude-sonnet-5", "medium");
	assert!(fields.get("thinking").is_some(), "got {fields}");
	assert!(fields.get("reasoning").is_none(), "got {fields}");
}

#[test]
fn test_responses_replayed_assistant_message_without_status_or_annotations_is_accepted() {
	// Codex CLI's second turn, captured 2026-09-04: the assistant message it replays carries
	// neither `status` nor `annotations`, which the typed OutputMessage requires.
	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "global.openai.gpt-5.6-sol",
		"max_output_tokens": 64,
		"tools": [{
			"type": "function",
			"name": "exec_command",
			"description": "Run a command",
			"parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
			"strict": false
		}],
		"input": [
			{"type": "message", "id": "msg_1", "role": "developer", "content": [{"type": "input_text", "text": "Be brief."}]},
			{"type": "message", "id": "msg_2", "role": "user", "content": [{"type": "input_text", "text": "How many entries in harbor?"}]},
			{"type": "function_call", "id": "call_1", "call_id": "call_1", "name": "exec_command", "arguments": "{\"cmd\":\"ls harbor | wc -l\"}"},
			{"type": "message", "id": "msg_3", "role": "assistant", "content": [{"type": "output_text", "text": "I'll list harbor and count the entries."}]},
			{"type": "function_call_output", "id": "fco_1", "call_id": "call_1", "output": "6\n"}
		]
	}))
	.expect("valid responses request");

	let body = super::from_responses::translate(&req, &reasoning_test_provider(), None, None, None)
		.expect("replayed assistant message must translate")
		.body;
	let translated: serde_json::Value = serde_json::from_slice(&body).unwrap();
	let rendered = translated.to_string();
	assert!(
		rendered.contains("I'll list harbor and count the entries."),
		"got {rendered}"
	);
	assert!(rendered.contains("toolUse"), "got {rendered}");
	assert!(rendered.contains("toolResult"), "got {rendered}");
}

const REDACTED_REASONING_BLOB: &str = "AQIDBAUGBwgJCgsMDQ4PEBESExQVFhcYGRobHB0eHyA=";

fn converse_stream_body(events: &[(&str, serde_json::Value)]) -> axum_core::body::Body {
	use aws_smithy_eventstream::frame::write_message_to;
	use aws_smithy_types::event_stream::{Header, HeaderValue, Message};

	let mut encoded = bytes::BytesMut::new();
	for (event_type, payload) in events {
		let message = Message::new(Bytes::from(serde_json::to_vec(payload).unwrap()))
			.add_header(Header::new(
				":message-type",
				HeaderValue::String("event".to_string().into()),
			))
			.add_header(Header::new(
				":event-type",
				HeaderValue::String(event_type.to_string().into()),
			));
		write_message_to(&message, &mut encoded).expect("event must encode");
	}
	axum_core::body::Body::from(encoded.freeze())
}

#[test]
fn test_responses_redacted_reasoning_becomes_reasoning_item() {
	// OpenAI models on Bedrock (gpt-6-astra, gpt-5.6-sol) return their reasoning only as an
	// encrypted redactedContent blob. It must surface as a Responses `reasoning` item carrying
	// `encrypted_content`, never as "[REDACTED]" text inside the message.
	let bytes = Bytes::from(
		serde_json::to_vec(&json!({
			"output": {"message": {"role": "assistant", "content": [
				{"reasoningContent": {"redactedContent": REDACTED_REASONING_BLOB}},
				{"text": "0.05"}
			]}},
			"stopReason": "end_turn",
			"usage": {"inputTokens": 40, "outputTokens": 120, "totalTokens": 160}
		}))
		.unwrap(),
	);
	let response =
		super::from_responses::translate_response(&bytes, "global.openai.gpt-6-astra", None)
			.expect("redacted reasoning response must translate");
	let rendered: serde_json::Value = serde_json::from_slice(&response.serialize().unwrap()).unwrap();
	let output = rendered["output"].as_array().expect("output array");
	assert_eq!(output.len(), 2, "got {rendered}");
	assert_eq!(output[0]["type"], "reasoning", "got {rendered}");
	assert!(
		output[0]["id"]
			.as_str()
			.unwrap_or_default()
			.starts_with("rs_"),
		"got {rendered}"
	);
	assert_eq!(output[0]["summary"], json!([]), "got {rendered}");
	assert_eq!(
		output[0]["encrypted_content"], REDACTED_REASONING_BLOB,
		"got {rendered}"
	);
	assert_eq!(output[1]["type"], "message", "got {rendered}");
	assert_eq!(output[1]["content"].as_array().map(Vec::len), Some(1));
	assert_eq!(output[1]["content"][0]["text"], "0.05", "got {rendered}");
	assert!(
		!rendered.to_string().contains("[REDACTED]"),
		"got {rendered}"
	);
}

#[tokio::test]
async fn test_responses_stream_redacted_reasoning_becomes_reasoning_item() {
	// Streaming twin of the buffered case: the redactedContent delta opens a reasoning item at
	// output_index 0 whose done event carries `encrypted_content`; the message item shifts to
	// output_index 1 and its text excludes any "[REDACTED]" marker.
	let body = converse_stream_body(&[
		("messageStart", json!({"role": "assistant"})),
		(
			"contentBlockDelta",
			json!({"contentBlockIndex": 0, "delta": {"reasoningContent": {"redactedContent": REDACTED_REASONING_BLOB}}}),
		),
		("contentBlockStop", json!({"contentBlockIndex": 0})),
		(
			"contentBlockDelta",
			json!({"contentBlockIndex": 1, "delta": {"text": "0."}}),
		),
		(
			"contentBlockDelta",
			json!({"contentBlockIndex": 1, "delta": {"text": "05"}}),
		),
		("contentBlockStop", json!({"contentBlockIndex": 1})),
		("messageStop", json!({"stopReason": "end_turn"})),
		(
			"metadata",
			json!({"usage": {"inputTokens": 40, "outputTokens": 120, "totalTokens": 160}}),
		),
	]);

	let out = super::from_responses::translate_stream(
		body,
		1024 * 1024,
		crate::StreamingUsageGuard::default(),
		"global.openai.gpt-6-astra",
		"req_1",
		crate::LogContentFields::default(),
		None,
	)
	.collect()
	.await
	.unwrap()
	.to_bytes();
	let raw = String::from_utf8(out.to_vec()).unwrap();
	assert!(!raw.contains("[REDACTED]"), "got {raw}");

	let events: Vec<serde_json::Value> = raw
		.lines()
		.filter_map(|line| line.strip_prefix("data: "))
		.map(|data| serde_json::from_str(data).expect("event json"))
		.collect();
	let sequence: Vec<u64> = events
		.iter()
		.map(|e| e["sequence_number"].as_u64().unwrap())
		.collect();
	assert!(
		sequence.windows(2).all(|w| w[0] < w[1]),
		"sequence numbers must be monotonic: {sequence:?}"
	);

	let reasoning_added = events
		.iter()
		.find(|e| e["type"] == "response.output_item.added" && e["item"]["type"] == "reasoning")
		.unwrap_or_else(|| panic!("no reasoning output_item.added in {raw}"));
	assert_eq!(reasoning_added["output_index"], 0);
	let reasoning_done = events
		.iter()
		.find(|e| e["type"] == "response.output_item.done" && e["item"]["type"] == "reasoning")
		.unwrap_or_else(|| panic!("no reasoning output_item.done in {raw}"));
	assert_eq!(reasoning_done["output_index"], 0);
	assert_eq!(
		reasoning_done["item"]["encrypted_content"],
		REDACTED_REASONING_BLOB
	);
	assert_eq!(reasoning_done["item"]["summary"], json!([]));
	assert_eq!(reasoning_done["item"]["id"], reasoning_added["item"]["id"]);
	assert!(
		reasoning_done["item"]["id"]
			.as_str()
			.unwrap_or_default()
			.starts_with("rs_")
	);

	let message_added = events
		.iter()
		.find(|e| e["type"] == "response.output_item.added" && e["item"]["type"] == "message")
		.unwrap_or_else(|| panic!("no message output_item.added in {raw}"));
	assert_eq!(message_added["output_index"], 1);
	let text_done = events
		.iter()
		.find(|e| e["type"] == "response.output_text.done")
		.unwrap_or_else(|| panic!("no output_text.done in {raw}"));
	assert_eq!(text_done["text"], "0.05");
	assert_eq!(text_done["output_index"], 1);
	let message_done = events
		.iter()
		.find(|e| e["type"] == "response.output_item.done" && e["item"]["type"] == "message")
		.unwrap_or_else(|| panic!("no message output_item.done in {raw}"));
	assert_eq!(message_done["output_index"], 1);
	assert_eq!(message_done["item"]["content"][0]["text"], "0.05");
}

#[test]
fn test_responses_replayed_reasoning_item_becomes_redacted_reasoning_block() {
	// Codex replays the previous turn's reasoning item (with `encrypted_content`) ahead of the
	// assistant message and function_call. It must land as a redactedContent reasoningContent
	// block at the head of the same Converse assistant message.
	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "global.openai.gpt-6-astra",
		"max_output_tokens": 64,
		"tools": [{
			"type": "function",
			"name": "exec_command",
			"description": "Run a command",
			"parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
			"strict": false
		}],
		"input": [
			{"type": "message", "id": "msg_1", "role": "user", "content": [{"type": "input_text", "text": "How many entries in harbor?"}]},
			{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": REDACTED_REASONING_BLOB},
			{"type": "message", "id": "msg_2", "role": "assistant", "content": [{"type": "output_text", "text": "I'll count them."}]},
			{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "exec_command", "arguments": "{\"cmd\":\"ls harbor | wc -l\"}"},
			{"type": "function_call_output", "id": "fco_1", "call_id": "call_1", "output": "6\n"}
		]
	}))
	.expect("valid responses request");

	let body = super::from_responses::translate(&req, &reasoning_test_provider(), None, None, None)
		.expect("replayed reasoning item must translate")
		.body;
	let translated: serde_json::Value = serde_json::from_slice(&body).unwrap();
	let messages = translated["messages"].as_array().expect("messages");
	assert_eq!(messages.len(), 3, "got {translated}");
	assert_eq!(messages[1]["role"], "assistant", "got {translated}");
	let assistant_content = messages[1]["content"]
		.as_array()
		.expect("assistant content");
	assert_eq!(
		assistant_content[0],
		json!({"reasoningContent": {"redactedContent": REDACTED_REASONING_BLOB}}),
		"got {translated}"
	);
	assert_eq!(
		assistant_content[1],
		json!({"text": "I'll count them."}),
		"got {translated}"
	);
	assert!(
		assistant_content[2].get("toolUse").is_some(),
		"got {translated}"
	);
	assert!(
		messages[2]["content"][0].get("toolResult").is_some(),
		"got {translated}"
	);
}

#[test]
fn test_responses_replayed_reasoning_item_without_encrypted_content_is_dropped() {
	// A summary-only reasoning item (what Anthropic reasoning is emitted as) has nothing Bedrock
	// can verify on replay, so it must not produce a reasoningContent block.
	let req: types::responses::Request = serde_json::from_value(json!({
		"model": "global.openai.gpt-6-astra",
		"max_output_tokens": 64,
		"input": [
			{"type": "message", "id": "msg_1", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
			{"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking"}]},
			{"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": ""},
			{"type": "message", "id": "msg_2", "role": "assistant", "content": [{"type": "output_text", "text": "Hello."}]},
			{"type": "message", "id": "msg_3", "role": "user", "content": [{"type": "input_text", "text": "Again"}]}
		]
	}))
	.expect("valid responses request");

	let body = super::from_responses::translate(&req, &reasoning_test_provider(), None, None, None)
		.expect("summary-only reasoning item must not fail the request")
		.body;
	let translated: serde_json::Value = serde_json::from_slice(&body).unwrap();
	assert!(
		!translated.to_string().contains("reasoningContent"),
		"got {translated}"
	);
	assert_eq!(
		translated["messages"][1]["content"],
		json!([{"text": "Hello."}]),
		"got {translated}"
	);
}
