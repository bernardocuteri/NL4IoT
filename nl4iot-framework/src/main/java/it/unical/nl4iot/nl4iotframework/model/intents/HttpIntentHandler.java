package it.unical.nl4iot.nl4iotframework.model.intents;

import java.util.Map;

public class HttpIntentHandler extends IntentHandler {

	public final static String HTTP_INTENT_HANDLER_TYPE = "HttpIntentHandler";
	
	private String baseUrl;

	private String method;

	private Map<String, String> mappings;

	private String bodyTemplate;

	private String urlTemplate;

	public HttpIntentHandler(String intent) {
		super(intent, HTTP_INTENT_HANDLER_TYPE);
	}

	public HttpIntentHandler(String intent, String baseUrl, String method, Map<String, String> mappings,
			String bodyTemplate, String urlTemplate) {
		super(intent, HTTP_INTENT_HANDLER_TYPE);
		this.baseUrl = baseUrl;
		this.method = method;
		this.mappings = mappings;
		this.bodyTemplate = bodyTemplate;
		this.urlTemplate = urlTemplate;
	}

	public String getBaseUrl() {
		return baseUrl;
	}

	public void setBaseUrl(String baseUrl) {
		this.baseUrl = baseUrl;
	}

	public String getMethod() {
		return method;
	}

	public void setMethod(String method) {
		this.method = method;
	}

	public Map<String, String> getMappings() {
		return mappings;
	}

	public void setMappings(Map<String, String> mappings) {
		this.mappings = mappings;
	}

	public String getBodyTemplate() {
		return bodyTemplate;
	}

	public void setBodyTemplate(String bodyTemplate) {
		this.bodyTemplate = bodyTemplate;
	}

	public String getUrlTemplate() {
		return urlTemplate;
	}

	public void setUrlTemplate(String urlTemplate) {
		this.urlTemplate = urlTemplate;
	}

}
